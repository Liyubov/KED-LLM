"""
Event-clustering pipeline
─────────────────────────
- Reads:
    – entities.csv            (nodes)
    – events2012_data_20k.csv (relations)
    – data_20k.csv            (column "text"; index = chunk_id)

- Builds an entity graph -> meta-graph of texts (node = chunk).
- Supports Louvain / Leiden / other algorithms
  and outputs AMI / ARI / NMI for each weekly slice
  + average metrics.

"""

import os, re, logging, functools, math, time
from pathlib import Path
from datetime import timedelta
from collections import defaultdict, Counter
from typing import Optional, Dict, Any


import optuna
import numpy as np
import pandas as pd
import networkx as nx
import torch
from tqdm.auto import tqdm
from dateutil.relativedelta import relativedelta

import community as community_louvain
from networkx.algorithms import community as nx_comm
import igraph as ig, leidenalg
from sklearn.cluster import KMeans
import hdbscan
from sklearn.metrics import (
    adjusted_mutual_info_score, adjusted_rand_score,
    normalized_mutual_info_score,
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from rapidfuzz import fuzz
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import faiss

os.environ.setdefault("OMP_NUM_THREADS", "4")

torch.set_num_threads(4)
torch.set_num_interop_threads(4)
try:
    faiss.omp_set_num_threads(4)
except AttributeError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

@functools.lru_cache(maxsize=None)
def _load_text_encoder(model_name: str, device: str | None = None):
    """Load SentenceTransformer model with a TF-IDF fallback."""
    try:
        return SentenceTransformer(model_name, device=device)
    except Exception as exc:
        logger.warning("Falling back to TF-IDF embeddings: %s", exc)

        class _TfidfEncoder:
            def __init__(self):
                self.vectorizer = TfidfVectorizer(max_features=512)

            def encode(self, texts, **kw):
                if not hasattr(self, "_fitted"):
                    self.vectorizer.fit(texts)
                    self._fitted = True
                return self.vectorizer.transform(texts).toarray()

        return _TfidfEncoder()

def _sanitize_attrs(d: dict):
    to_del = []
    for k, v in d.items():
        if v is None or (isinstance(v, float) and math.isnan(v)) \
           or (pd.isna(v) if not isinstance(v, str) else False):
            d[k] = ""
            continue
        if isinstance(v, np.generic):
            d[k] = v.item()
        elif isinstance(v, (np.ndarray, list, tuple, set, dict)):
            d[k] = str(v)


def prepare_for_export(G: nx.Graph) -> nx.Graph:
    G2 = G.copy()
    for _, nd in G2.nodes(data=True):
        _sanitize_attrs(nd)
    for _, _, ed in G2.edges(data=True):
        _sanitize_attrs(ed)
    return G2


def save_graph(G: nx.Graph | nx.DiGraph, path: str | Path, fmt: str = "graphml"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    G_exp = prepare_for_export(G)

    if fmt == "graphml":
        nx.write_graphml(G_exp, path)
    elif fmt == "gexf":
        nx.write_gexf(G_exp, path)
    else:
        raise ValueError(f"Unknown format {fmt}")

_url_pat  = re.compile(r"https?://\S+|www\.\S+")
_hash_pat = re.compile(r"#\w+")

def clean_text(txt: str | float) -> str:
    if pd.isna(txt):
        return ""
    txt = _url_pat.sub(" ", txt)
    txt = _hash_pat.sub(" ", txt)
    return re.sub(r"\s+", " ", txt).strip()


def load_texts(texts_csv: str, text_col: str = "text") -> dict[int, str]:
    df = pd.read_csv(texts_csv, usecols=[text_col])
    df["clean"] = df[text_col].apply(clean_text)
    return df["clean"].to_dict()

def normalize_label(label) -> str | None:
    if pd.isna(label):
        return None
    return str(label).lower().lstrip("#")


def load_data(nodes_csv: str, relations_csv: str):
    return pd.read_csv(nodes_csv), pd.read_csv(relations_csv)


def split_multiword_nodes(G: nx.DiGraph) -> nx.DiGraph:
    multi = [n for n in G if " " in n]
    for node in multi:
        node_attrs = G.nodes[node]
        base_type  = node_attrs.get("type")

        toks = node.split()
        for t in toks:
            if not G.has_node(t):
                G.add_node(
                    t,
                    type=base_type,
                    chunk=node_attrs.get("chunk"),
                    original_label=t,
                )

        for i in range(len(toks) - 1):
            G.add_edge(toks[i], toks[i + 1], label="related")

        for pred in list(G.predecessors(node)):
            G.add_edge(pred, toks[0], **G[pred][node])
        for succ in list(G.successors(node)):
            G.add_edge(toks[-1], succ, **G[node][succ])

        G.remove_node(node)
    return G


def merge_similar_nodes(G: nx.Graph, threshold=90, method="token_sort"):
    labels = list(G)
    mapping, seen = {}, set()
    sim_func = fuzz.token_sort_ratio if method == "token_sort" else fuzz.token_set_ratio
    for i, l in enumerate(labels):
        if l in seen:
            continue
        seen.add(l)
        group = [l]
        for other in labels[i + 1:]:
            if other in seen:
                continue
            if sim_func(l, other) >= threshold:
                seen.add(other)
                group.append(other)
        if len(group) > 1:
            rep = min(group, key=len)
            for n in group:
                if n != rep:
                    mapping[n] = rep
    if mapping:
        nx.relabel_nodes(G, mapping, copy=False)
    return G

def add_semantic_type_edges_faiss(
        G: nx.Graph,
        *,
        include_types : list[str] | None,
        sim_thresholds: dict[str, float] | None,
        default_thr   : float,
        alpha         : float | dict[str, float],
        model_name    : str,
        device        : str | None = None,
        fp16          : bool = True,
        top_k         : int = 256,
):
    """
    Quickly adds edges within each type using FAISS (cosine/IP).

    - One-shot encoding of all labels;
    - A single shared FAISS index (CPU or GPU) → top-k searches;
    - Filter sim more or equal to threshold for each type.
    """

    include_types = set(include_types) if include_types else None
    sim_thresholds = sim_thresholds or {}

    nodes, labels, types = [], [], []
    for n, d in G.nodes(data=True):
        t = d.get('type')
        if t and (include_types is None or t in include_types):
            nodes.append(n)
            labels.append(d.get('original_label', n))
            types.append(t)
    if len(nodes) < 2:
        return G

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_text_encoder(model_name, device=device)
    vecs = model.encode(labels, convert_to_numpy=True,
                        batch_size=1024, show_progress_bar=False)
    if fp16:
        vecs32 = vecs.astype('float32', copy=False)
        faiss.normalize_L2(vecs32)
        if fp16:
            vecs = vecs32.astype('float16')
        else:
            vecs = vecs32
    else:
        faiss.normalize_L2(vecs)

    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    if device == "cuda" and faiss.get_num_gpus() > 0:
        res   = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(vecs)

    D, I = index.search(vecs, top_k)   # (N, k)
    for i, (idxs, sims) in enumerate(zip(I, D)):
        t_i   = types[i]
        thr_i = sim_thresholds.get(t_i, default_thr)
        alpha_i = alpha[t_i] if isinstance(alpha, dict) else alpha
        u = nodes[i]
        for j, s in zip(idxs[1:], sims[1:]):     
            if s < thr_i or j < 0:
                continue
            v = nodes[j]
            if types[j] != t_i:
                continue
            if G.degree(u) > 30 or G.degree(v) > 30:
                continue
            w_raw = s * alpha_i
            hub_factor = math.log1p(G.degree(u) * G.degree(v))
            w_final = w_raw / hub_factor
            w_old = G[u][v].get('weight', 0) if G.has_edge(u, v) else 0
            G.add_edge(
                u,
                v,
                weight=w_old + w_final,
                edge_type=f"sem_{t_i}",
            )
    return G


def add_cross_type_edges_faiss(
        G: nx.Graph,
        *,
        type_pairs : list[tuple[str, str]],
        sim_threshold : float | dict[tuple[str, str], float] = 0.75,
        alpha         : float | dict[tuple[str, str], float] = 0.5,
        model_name    : str = "sentence-transformers/paraphrase-MiniLM-L12-v2",
        device        : str | None = None,
        fp16          : bool = True,
        top_k         : int = 128,
    ) -> nx.Graph:
    """Adds semantic edges between types.

    type_pairs - list of allowed type pairs (order-insensitive).
    sim_threshold - cosine similarity threshold.
    alpha - weight multiplier for such edges.
    """
    if not type_pairs:
        return G

    allowed = {tuple(sorted(p)) for p in type_pairs}

    nodes, labels, types = [], [], []
    for n, d in G.nodes(data=True):
        t = d.get('type')
        if t and any(t in p for p in allowed):
            nodes.append(n)
            labels.append(d.get('original_label', n))
            types.append(t)
    if len(nodes) < 2:
        return G

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_text_encoder(model_name, device=device)
    vecs = model.encode(labels, convert_to_numpy=True,
                        batch_size=1024, show_progress_bar=False)
    if fp16:
        vecs32 = vecs.astype('float32', copy=False)
        faiss.normalize_L2(vecs32)
        vecs = vecs32.astype('float16')
    else:
        faiss.normalize_L2(vecs)

    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    if device == "cuda" and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(vecs)

    D, I = index.search(vecs, top_k)
    for i, (idxs, sims) in enumerate(zip(I, D)):
        u = nodes[i]
        t_i = types[i]
        for j, s in zip(idxs[1:], sims[1:]):
            if j < 0:
                continue
            v = nodes[j]
            t_j = types[j]
            pair = tuple(sorted((t_i, t_j)))
            thr_pair = sim_threshold.get(pair, 0.75) if isinstance(sim_threshold, dict) else sim_threshold
            if s < thr_pair or pair not in allowed:
                continue
            if G.degree(u) > 30 or G.degree(v) > 30:
                continue
            alpha_pair = alpha[pair] if isinstance(alpha, dict) else alpha
            w_raw = s * alpha_pair
            hub_factor = math.log1p(G.degree(u) * G.degree(v))
            w_final = w_raw / hub_factor
            w_old = G[u][v].get('weight', 0) if G.has_edge(u, v) else 0
            etype = f"cross_{t_i}_{t_j}"
            G.add_edge(u, v, weight=w_old + w_final, edge_type=etype)
    return G


def geocode_toponyms(toponyms: list[str], *, user_agent: str = "ked_geocoder",
                     delay: float = 1.0, cache_file: str | None = None
    ) -> dict[str, tuple[float, float]]:
    """Geocodes a list of toponyms using Nominatim.

    Results can be cached in cache_file (a CSV with columns
    toponym, lat, lon). Subsequent calls read the cache and update it.
    """
    cache: dict[str, tuple[float, float]] = {}
    if cache_file and os.path.exists(cache_file):
        df = pd.read_csv(cache_file)
        cache = {r['toponym']: (r['lat'], r['lon']) for _, r in df.iterrows()}

    geo = Nominatim(user_agent=user_agent)
    coords = {}
    for t in toponyms:
        if t in cache:
            coords[t] = cache[t]
            continue
        try:
            loc = geo.geocode(t, language="en", exactly_one=True, timeout=10)
            if loc:
                coords[t] = (loc.latitude, loc.longitude)
                cache[t] = coords[t]
        except Exception:
            continue
        if delay:
            time.sleep(delay)

    if cache_file:
        rows = [{"toponym": k, "lat": v[0], "lon": v[1]} for k, v in cache.items()]
        pd.DataFrame(rows).to_csv(cache_file, index=False)
    return coords


def add_geo_edges(
        G: nx.Graph,
        coords: dict[str, tuple[float, float]],
        *,
        dist_thr: float = 500.0,
        alpha: float = 1.0
    ) -> nx.Graph:
    """Adds geo-edges between nodes of type toponym.

    dist_thr - maximum distance (km) for adding an edge.
    Edge weight is inversely proportional to distance:
    (1 - d_km / dist_thr) * alpha.
    """
    nodes = [n for n, d in G.nodes(data=True)
             if d.get("type") == "toponym" and n in coords]
    for i, u in enumerate(nodes):
        for v in nodes[i + 1:]:
            if G.degree(u) > 30 or G.degree(v) > 30:
                continue
            d_km = geodesic(coords[u], coords[v]).kilometers
            if d_km > dist_thr:
                continue
            w_raw = (1.0 - d_km / dist_thr) * alpha
            hub_factor = math.log1p(G.degree(u) * G.degree(v))
            w_new = w_raw / hub_factor
            w_old = G[u][v].get("weight", 0) if G.has_edge(u, v) else 0
            G.add_edge(u, v,
                       weight=w_old + w_new,
                       edge_type="geo",
                       distance_km=d_km)
    return G


def build_graph(
        nodes: pd.DataFrame,
        relations: pd.DataFrame,
        *,
        do_split     : bool = True,     
        do_merge     : bool = True,     
        merge_thr    : int  = 95,      
):
    """
    - Builds a base directed entity graph only from relations.  
    - (opt.) Splits multi-word nodes into tokens
    - (opt.) Merges similar labels using RapidFuzz
    """
    t_map, c_map = {}, {}
    for _, row in nodes.iterrows():
        ent = normalize_label(row['entity'])
        if ent:
            t_map[ent] = row.get('type')
            c_map[ent] = row.get('chunk')

    G = nx.DiGraph()

    for _, row in tqdm(relations.iterrows(), total=len(relations)):
        u_raw, v_raw = row['subject'], row['object']
        u, v = normalize_label(u_raw), normalize_label(v_raw)
        if not u or not v:
            continue

        for raw, norm in ((u_raw, u), (v_raw, v)):
            if not G.has_node(norm):
                G.add_node(
                    norm,
                    type=t_map.get(norm),
                    chunk=c_map.get(norm),
                    original_label=raw,
                )

        G.add_edge(
            u, v,
            label       = row.get('predicate'),
            time        = row.get('time'),
            source_text = row.get('source_text'),
            event_id    = row.get('event_id'),
            weight      = 1.0,
            edge_type   = "entity",
        )

    if do_split:
        G = split_multiword_nodes(G)
    if do_merge:
        G = merge_similar_nodes(G, threshold=merge_thr)

    return G

def slice_graph_by_time(G: nx.DiGraph, interval='week', time_attr='time'):
    times = pd.to_datetime([d[time_attr] for _,_,d in G.edges(data=True) if d.get(time_attr)])
    if times.empty:
        return {}
    start, end = times.min().normalize(), times.max().normalize()

    step = {'day': timedelta(days=1),
            'week': timedelta(weeks=1),
            'month': relativedelta(months=1)}[interval]

    slices, idx, cursor = {}, 0, start
    while cursor <= end:
        nxt = cursor + step
        edges = [(u,v) for u,v,d in G.edges(data=True)
                 if d.get(time_attr) and cursor <= pd.to_datetime(d[time_attr]) < nxt]
        slices[f'slice_{idx}_{interval}'] = G.edge_subgraph(edges).copy()
        cursor, idx = nxt, idx+1
    return slices


def _parse_period(p):
    if isinstance(p, str):
        if p.endswith('d'):
            return timedelta(days=int(p[:-1]))
        if p.endswith('w'):
            return timedelta(weeks=int(p[:-1]))
        if p.endswith('m'):
            return relativedelta(months=int(p[:-1]))
    return p


def slice_graph_sliding(G: nx.DiGraph,
                        *,
                        window: str | timedelta = '2w',
                        step  : str | timedelta = '1w',
                        time_attr='time') -> dict[str, nx.DiGraph]:
    """Slices the graph with a sliding window."""
    times = pd.to_datetime([d[time_attr] for _,_,d in G.edges(data=True) if d.get(time_attr)])
    if times.empty:
        return {}
    start, end = times.min().normalize(), times.max().normalize()
    win  = _parse_period(window)
    step = _parse_period(step)

    slices, idx, cur = {}, 0, start
    while cur <= end:
        nxt = cur + win
        edges = [(u,v) for u,v,d in G.edges(data=True)
                 if d.get(time_attr) and cur <= pd.to_datetime(d[time_attr]) < nxt]
        slices[f'slice_{idx}_sliding'] = G.edge_subgraph(edges).copy()
        cur += step
        idx += 1
    return slices

_txt_emb_cache: dict[int, np.ndarray] = {}
_n2v_cache: dict[tuple, dict[str, np.ndarray]] = {}

def get_chunk_embeddings(
    text_map: dict[int, str],
    model_name: str = "sentence-transformers/paraphrase-MiniLM-L12-v2",
    batch_size: int = 64,
) -> dict[int, np.ndarray]:
    if not text_map:
        return {}
    model = _load_text_encoder(model_name)
    to_encode: list[tuple[int, str]] = []
    out: dict[int, np.ndarray] = {}
    for cid, txt in text_map.items():
        if cid in _txt_emb_cache:
            out[cid] = _txt_emb_cache[cid]
        else:
            to_encode.append((cid, txt))
    if to_encode:
        ids, txts = zip(*to_encode)
        vecs = model.encode(list(txts), batch_size=batch_size,
                            show_progress_bar=False, convert_to_numpy=True)
        for cid, vec in zip(ids, vecs):
            _txt_emb_cache[cid] = vec
            out[cid] = vec
    return out


def node2vec_embeddings(
        G: nx.Graph,
        *,
        dimensions: int = 64,
        walk_length: int = 30,
        num_walks: int = 200,
        p: float = 1.0,
        q: float = 1.0,
        workers: int = 4,
        epochs: int = 1,
        seed: int = 42) -> dict[str, np.ndarray]:
    """Computes node2vec embeddings for G.

    Requires the node2vec package.
    Returns mapping node -> embedding.
    """
    try:
        from node2vec import Node2Vec  
    except Exception as exc: 
        raise ImportError("node2vec package is required for node2vec_embeddings") from exc

    Gu = G if not G.is_directed() else G.to_undirected()
    key = (
        tuple(sorted(Gu.edges())),
        dimensions,
        walk_length,
        num_walks,
        p,
        q,
        epochs,
        seed,
    )
    if key in _n2v_cache:
        return _n2v_cache[key]

    n2v = Node2Vec(
        Gu,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        workers=workers,
        seed=seed,
    )
    model = n2v.fit(window=10, min_count=1, batch_words=4,
                    workers=workers, epochs=epochs)
    res = {str(n): model.wv[str(n)] for n in Gu.nodes()}
    _n2v_cache[key] = res
    return res


def _text_edge_weights(
    emb_map: dict[int, np.ndarray],
    *,
    sim_thr: float = 0.6,
    alpha: float,
) -> dict[tuple[int, int], float]:
    """Return pairwise text-similarity weights between chunk IDs."""
    if len(emb_map) < 2:
        return {}
    ids = list(emb_map)
    mat = cosine_similarity(np.stack([emb_map[i] for i in ids]))
    weights: dict[tuple[int, int], float] = {}
    n = len(ids)
    for i in range(n):
        for j in range(i + 1, n):
            sim = mat[i, j]
            if sim < sim_thr:
                continue
            u, v = ids[i], ids[j]
            weights[tuple(sorted((u, v)))] = sim * alpha
    return weights


def build_meta_graph(G: nx.DiGraph,
                     texts_map: dict[int,str] | None = None,
                     add_text_edges=True,
                     sim_thr=0.6,
                     text_alpha=None) -> nx.Graph:
    H = nx.Graph()
    chunks = {d['chunk'] for _, d in G.nodes(data=True) if d.get('chunk') is not None}
    for c in chunks:
        H.add_node(c, text=texts_map.get(c) if texts_map else None)


    pair_type_weight: dict[tuple[int, int], dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for u, v, d in G.edges(data=True):
        cu, cv = (G.nodes[u].get('chunk'), G.nodes[v].get('chunk'))
        if cu is None or cv is None or cu == cv:
            continue
        etype = d.get('edge_type', 'entity')
        w = d.get('weight', 1.0)
        if w <= 0:
            continue
        key = tuple(sorted((cu, cv)))
        pair_type_weight[key][etype] += w

    if add_text_edges and texts_map:
        emb_map = get_chunk_embeddings({c: texts_map[c] for c in chunks})
        if text_alpha is None:
            existing = [w for dct in pair_type_weight.values() for w in dct.values()]
            base = np.median(existing) if existing else 1.0
            text_alpha = 0.5 * base
        text_weights = _text_edge_weights(emb_map, sim_thr=sim_thr, alpha=text_alpha)
        for (cu, cv), w in text_weights.items():
            pair_type_weight[(cu, cv)]["text_sim"] += w

    weight_by_type: dict[str, list[float]] = defaultdict(list)
    for w_dict in pair_type_weight.values():
        for t, w in w_dict.items():
            weight_by_type[t].append(w)

    med = {t: np.median(ws) for t, ws in weight_by_type.items() if ws}

    for (cu, cv), w_dict in pair_type_weight.items():
        total = 0.0
        for t, w in w_dict.items():
            m = med.get(t, 1.0) or 1.0
            total += w / m
        H.add_edge(cu, cv, weight=total)

    return H

def detect_communities(G: nx.Graph, method='louvain', **kw) -> dict[int,set]:
    Gu = G if not G.is_directed() else G.to_undirected()

    if method == 'louvain':
        part = community_louvain.best_partition(Gu, **kw)
        comms = defaultdict(set)
        for n,c in part.items(): comms[c].add(n)
        return comms

    if method == 'leiden':
        edges = list(Gu.edges(data='weight', default=1))
        igG = ig.Graph.TupleList(edges, directed=False, edge_attrs=['weight'])
        part = leidenalg.find_partition(
            igG, leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=kw.get('resolution_parameter',1),
            weights='weight', seed=kw.get('random_state',42))
        comms = defaultdict(set)
        for idx, cid in enumerate(part.membership):
            comms[cid].add(igG.vs[idx]['name'])
        return comms

    if method == 'label_prop':
        comp = nx_comm.asyn_lpa_communities(Gu, seed=kw.get('random_state',42))
        return {i:set(c) for i,c in enumerate(comp)}

    if method == 'n2v_kmeans':
        emb = node2vec_embeddings(Gu, **kw.get('n2v_params', {}))
        X = np.stack([emb[str(n)] for n in Gu.nodes()])
        n_clusters = kw.get('n_clusters', max(2, int(math.sqrt(len(Gu)))))
        km = KMeans(n_clusters=n_clusters, random_state=kw.get('random_state', 42))
        labels = km.fit_predict(X)
        comms = defaultdict(set)
        for node, lbl in zip(Gu.nodes(), labels):
            comms[int(lbl)].add(node)
        return comms

    if method == 'n2v_hdbscan':
        emb = node2vec_embeddings(Gu, **kw.get('n2v_params', {}))
        X = np.stack([emb[str(n)] for n in Gu.nodes()])
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=kw.get('min_cluster_size', 5),
            min_samples=kw.get('min_samples', None),
            metric='euclidean',
        )
        labels = clusterer.fit_predict(X)
        comms = defaultdict(set)
        for node, lbl in zip(Gu.nodes(), labels):
            comms[int(lbl)].add(node)
        return comms

    raise ValueError(f'Unsupported method {method}')

def get_true_labels_by_chunk(G: nx.DiGraph) -> dict[int,int]:
    ctr = defaultdict(Counter)
    for u,v,d in G.edges(data=True):
        ev = d.get('event_id')
        if pd.isna(ev):
            continue
        for n in (u,v):
            c = G.nodes[n].get('chunk')
            if c is not None:
                ctr[c][ev] += 1
    return {c:cnt.most_common(1)[0][0] for c,cnt in ctr.items()}


def evaluate_meta_slice(H: nx.Graph,
                        comms: dict[int,set],
                        true_lbls: dict[int,int]) -> dict[str,float]:
    nodes = [n for n in H if n in true_lbls]
    if not nodes:
        return {'AMI':None,'ARI':None,'NMI':None}
    y_true = [true_lbls[n] for n in nodes]
    pred = {n:cid for cid,grp in comms.items() for n in grp}
    y_pred = [pred.get(n,-1) for n in nodes]
    return {
        'AMI': adjusted_mutual_info_score(y_true,y_pred),
        'ARI': adjusted_rand_score(y_true,y_pred),
        'NMI': normalized_mutual_info_score(y_true,y_pred)
    }

def run_meta_pipeline(
        nodes_csv      : str,
        relations_csv  : str,
        texts_csv      : str,
        *,
        interval            : str = "week",
        window              : Optional[str] = None,
        step                : Optional[str] = None,
        community_method    : str = "louvain",
        community_params    : Optional[Dict[str, Any]] = None,
        intra_type_edges    : Optional[Dict] = None,
        cross_type_edges    : Optional[Dict] = None,
        geo_edges           : Optional[Dict] = None,
        sim_thr             : float = 0.60,
        text_alpha          : Optional[float] = None,
        save_dir            : str | Path | None = None,
        save_format         : str = "graphml",
):
    """Run clustering pipeline.

    Geographic and semantic edges are injected after temporal slicing so that
    only surviving nodes form connections.
    """

    community_params = community_params or {}

    nodes_df, rels_df = load_data(nodes_csv, relations_csv)
    texts_map         = load_texts(texts_csv)

    G = build_graph(nodes_df, rels_df)

    if save_dir:
        save_graph(G, Path(save_dir) / f"00_entity_graph.{save_format}", save_format)

    coords = None
    if geo_edges:
        topo  = [n for n, d in G.nodes(data=True) if d.get("type") == "toponym"]
        cache = geo_edges.get("cache_file") if geo_edges else None
        coords = geocode_toponyms(
            topo,
            cache_file=cache,
            user_agent=(geo_edges or {}).get("user_agent", "ked"),
            delay=(geo_edges or {}).get("delay", 0.5),
        )

    if window:
        slices = slice_graph_sliding(G, window=window, step=step or window)
    else:
        slices = slice_graph_by_time(G, interval=interval)

    rows = []
    for sid, subG in slices.items():
        if intra_type_edges:
            if "sim_thresholds" not in intra_type_edges:
                intra_type_edges = {**intra_type_edges, "sim_thresholds": {}}
            subG = add_semantic_type_edges_faiss(subG, **intra_type_edges)
        if cross_type_edges:
            subG = add_cross_type_edges_faiss(subG, **cross_type_edges)
        if geo_edges and coords:
            subG = add_geo_edges(
                subG,
                coords,
                dist_thr=(geo_edges or {}).get("dist_thr", 500.0),
                alpha=(geo_edges or {}).get("alpha", 1.0),
            )

        if save_dir:
            save_graph(subG, Path(save_dir) / f"{sid}_01_enriched.{save_format}", save_format)

        H = build_meta_graph(
            subG,
            texts_map,
            add_text_edges=True,
            sim_thr=sim_thr,
            text_alpha=text_alpha,
        )

        if save_dir:
            save_graph(H, Path(save_dir) / f"{sid}_02_meta.{save_format}", save_format)

        true_lbl = get_true_labels_by_chunk(subG)
        comms    = detect_communities(H, community_method, **community_params)

        for cid, nodes in comms.items():
            for n in nodes:
                H.nodes[n]["community"] = cid
        for n, ev in true_lbl.items():
            if n in H:
                H.nodes[n]["event_id"] = ev

        if save_dir:
            save_graph(H, Path(save_dir) / f"{sid}_03_meta_comm.{save_format}", save_format)

        scores = evaluate_meta_slice(H, comms, true_lbl)
        rows.append({"slice": sid, **scores})

    df_metrics  = pd.DataFrame(rows)
    mean_scores = df_metrics[["AMI", "ARI", "NMI"]].mean().to_dict()

    param_summary = dict(
        window=window,
        step=step,
        interval=interval,
        community_method=community_method,
        community_params=community_params,
        intra_type_edges=intra_type_edges,
        cross_type_edges=cross_type_edges,
        geo_edges=geo_edges,
        sim_thr=sim_thr,
        text_alpha=text_alpha,
        save_dir=str(save_dir) if save_dir else None,
        save_format=save_format,
    )

    logger.info(f"Mean metrics {mean_scores}")

    return df_metrics, mean_scores, param_summary


def iterative_search_params(
        nodes_csv: str,
        relations_csv: str,
        texts_csv: str,
        *,
        method: str = 'louvain',
        gamma_range=(0.5, 2.0),
        sim_thr_range=(0.5, 0.7),
        text_alpha_range=(0.25, 1.0),
        interval: str = 'week',
        n_trials: int = 25,
        random_state: int = 42):
    """Bayesian optimisation maximising mean NMI.
    """
    use_n2v = method in {'n2v_kmeans', 'n2v_hdbscan'}

    def objective(trial: optuna.Trial) -> float:
        gamma   = trial.suggest_float('gamma', *gamma_range)
        thr = trial.suggest_float('sim_thr', *sim_thr_range)
        alpha   = trial.suggest_float('text_alpha', *text_alpha_range)

        if use_n2v:
            walk_len = trial.suggest_int('walk_length', 30, 60, step=10)
            p_val    = trial.suggest_float('p', 0.25, 4.0, log=True)
            q_val    = trial.suggest_float('q', 0.25, 4.0, log=True)

            n2v_params = {
                'dimensions': 128,
                'walk_length': walk_len,
                'num_walks': 400,
                'p': p_val,
                'q': q_val,
                'workers': 4,
                'epochs': 1,
                'seed': random_state,  
            }

            comm_params = {'n2v_params': n2v_params}

            if method == 'n2v_kmeans':
                comm_params.update({
                    'n_clusters': 16,
                    'random_state': random_state,  
                })
            else:  
                comm_params.update({
                    'min_cluster_size': trial.suggest_int('min_cluster_size', 5, 25, step=5),
                    'min_samples': trial.suggest_int('min_samples', 1, 10),
                })
        else:
            if method == 'louvain':
                comm_params = {'resolution': gamma, 'random_state': random_state}
            else: 
                comm_params = {'resolution_parameter': gamma, 'random_state': random_state}

        _, mean_scores, _ = run_meta_pipeline(
            nodes_csv=nodes_csv,
            relations_csv=relations_csv,
            texts_csv=texts_csv,
            interval=interval,
            community_method=method,
            community_params=comm_params,
            sim_thr=thr,
            text_alpha=alpha,
        )

        nmi = mean_scores['NMI']

        trial.set_user_attr('mean_AMI', mean_scores['AMI'])
        trial.set_user_attr('mean_ARI', mean_scores['ARI'])
        trial.set_user_attr('mean_NMI', nmi)
        return nmi

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = {
        **study.best_trial.params,
        'mean_AMI': study.best_trial.user_attrs['mean_AMI'],
        'mean_ARI': study.best_trial.user_attrs['mean_ARI'],
        'mean_NMI': study.best_trial.user_attrs['mean_NMI'],
    }
    return best, study.trials_dataframe()