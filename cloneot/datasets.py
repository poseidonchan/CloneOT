"""The two 2D datasets of the paper, and public lineage-tracing data.

In both 2D datasets, source and target are unpaired samples of two classes, and optimal transport with the squared
Euclidean cost sends points to the wrong class:

* ``t_shape``: the T-shape example of Howard et al. (2024). A vertical bar on the left moves to the right and a
  horizontal bar on top moves down. The points form corresponding subsets of two source and two target points at one
  position of a bar.
* ``rotated_moons``: the task of OT-SI (Liu et al., 2020). The target is the two moons rotated by 90 degrees, and the
  only lineage is which source moon corresponds to which rotated moon.

Each function returns a dictionary with the source and target points, their classes (``source_class``,
``target_class``), the labels of the corresponding subsets (``source_clone``, ``target_clone``, which play the role of
clones) and the true target position ``fate(x)`` of a source point.
"""
from __future__ import annotations

import numpy as np


def t_shape(n: int = 400, cells: int = 2, noise: float = 0.05, seed: int = 0) -> dict:
    """T-shape dataset: n source and n target points in corresponding subsets of `cells` points, half on each bar."""
    rng = np.random.default_rng(seed)
    shift = np.array([[14., 0.], [0., -10.]])
    bar = np.repeat([0, 1], n//cells//2)
    u = rng.uniform(size=len(bar))
    ancestor = np.where(bar[:, None] == 0, np.stack([np.full_like(u, -7), 3+2*u], 1),
                        np.stack([2*u-1, np.full_like(u, 5)], 1))
    clone = np.repeat(np.arange(len(bar)), cells)
    source = ancestor[clone]+noise*rng.normal(size=(len(clone), 2))
    target = ancestor[clone]+shift[bar[clone]]+noise*rng.normal(size=(len(clone), 2))
    return dict(source=source, target=target, source_class=bar[clone], target_class=bar[clone],
                source_clone=clone, target_clone=clone, fate=lambda x: x+shift[(x[:, 0] > -4).astype(int)])


def rotated_moons(n: int = 400, angle: float = 90., noise: float = 0.05, seed: int = 0) -> dict:
    """Rotated two moons: the target is an independent sample of the two moons rotated by `angle` degrees about their
    centre. The two moons are the corresponding subsets."""
    from sklearn.datasets import make_moons
    centre = np.array([.5, .25])
    a = np.deg2rad(angle)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    source, source_moon = make_moons(n_samples=n, noise=noise, random_state=seed)
    target, target_moon = make_moons(n_samples=n, noise=noise, random_state=seed+1000)
    return dict(source=source, target=(target-centre)@R.T+centre, source_class=source_moon, target_class=target_moon,
                source_clone=source_moon, target_clone=target_moon, fate=lambda x: (x-centre)@R.T+centre)


def label_fraction(source_clone, target_clone, fraction: float, seed: int = 0):
    """Keep the labels of a random `fraction` of the subsets (clones) and mark all other points as unlabeled (-1)."""
    clones = np.unique(source_clone)
    shown = np.random.default_rng(seed).permutation(clones)[:int(np.ceil(fraction*len(clones)))]
    index = {c: i for i, c in enumerate(sorted(shown.tolist()))}
    return [np.array([index.get(c, -1) for c in labels]) for labels in (source_clone, target_clone)]


def class_accuracy(prediction, true_class, reference: dict) -> float:
    """Fraction of predictions whose nearest reference target cell belongs to the true class of the source cell."""
    nearest = ((prediction[:, None]-reference['target'][None])**2).sum(-1).argmin(1)
    return float((reference['target_class'][nearest] == true_class).mean())


# --------------------------------------------------------------------------------------------- lineage-tracing data
GEO = 'https://ftp.ncbi.nlm.nih.gov/geo/series'
WATERMELON_FILES = {
    'GSE150949_metaData_with_lineage.txt.gz': '91da44b34714a1bcbb7f37b92aa371d55e53b09df42be7c89191223ebab6b1f4',
    'GSE150949_pc9_count_matrix.csv.gz': '37b4e957392121d5ab8c8475793415d58dfc881eab5a0eb736773b6b464febce',
}
FRACTIONS = {'high': 'non-cycling', 'med': 'moderate', 'low': 'cycling'}


def _sha256(path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def download(url: str, path, sha256: str | None = None) -> None:
    """Download `url` to `path` (skipped if a file with the expected checksum exists)."""
    import shutil
    import urllib.request
    from pathlib import Path
    path = Path(path)
    if path.exists() and (sha256 is None or _sha256(path) == sha256):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix+'.part')
    print(f'Downloading {url}', flush=True)
    with urllib.request.urlopen(url, timeout=120) as response, open(part, 'wb') as out:
        shutil.copyfileobj(response, out)
    if sha256 is not None and _sha256(part) != sha256:
        part.unlink()
        raise IOError(f'checksum mismatch for {url}')
    part.rename(path)


def _single_barcode_clones(raw):
    """Clone labels of cells with exactly one lineage barcode (cells with several barcodes get no label, since the
    barcodes may come from multiple integrations or doublets) and lineage blocks that join clones sharing a barcode."""
    import pandas as pd
    strings = raw.fillna('').astype(str).replace('NA', '')
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for value in sorted(set(strings)-{''}):
        parts = sorted(set(value.split(',')))
        for atom in parts:
            a, b = find(parts[0]), find(atom)
            if a != b:
                parent[max(a, b)] = min(a, b)
    blocks = strings.map(lambda s: find(s.split(',')[0]) if s else '')
    clones = strings.where(~strings.str.contains(','), '')
    return clones.replace('', pd.NA), blocks.replace('', pd.NA)


def watermelon(data_dir: str = 'data', download_files: bool = True):
    """Watermelon lineage tracing of PC9 lung cancer cells treated with osimertinib (Oren et al., Nature 2021;
    GEO GSE150949).

    Cells were profiled before treatment (day 0) and at days 3, 7 and 14; day-14 persister cells were sorted by a
    proliferation-history reporter into cycling, moderate and non-cycling fractions. The first call downloads 125 MB
    from GEO and caches ``<data_dir>/watermelon.h5ad``.

    Returns
    -------
    AnnData with raw counts in ``X`` and ``obs`` columns ``clone`` (single-barcode clone label, NA otherwise),
    ``lineage_block``, ``day``, ``condition`` ('day0', 'day3', 'day7', 'day14'), ``fraction`` ('cycling', 'moderate',
    'non-cycling' or 'unsorted' for day 14; 'unsorted' before), ``sample`` and ``replicate``.
    """
    import csv
    import gzip
    from pathlib import Path

    import anndata as ad
    import pandas as pd
    from scipy import sparse
    folder = Path(data_dir)
    cache = folder/'watermelon.h5ad'
    if cache.exists():
        return ad.read_h5ad(cache)
    raw = folder/'GSE150949'
    for name, checksum in WATERMELON_FILES.items():
        if download_files:
            download(f'{GEO}/GSE150nnn/GSE150949/suppl/{name}', raw/name, checksum)
        elif not (raw/name).exists():
            raise FileNotFoundError(raw/name)
    metadata = pd.read_csv(raw/'GSE150949_metaData_with_lineage.txt.gz', sep='\t')
    genes, indptr, indices, values = [], [0], [], []
    with gzip.open(raw/'GSE150949_pc9_count_matrix.csv.gz', 'rt') as handle:     # genes x cells, streamed
        cells = next(csv.reader([handle.readline()]))[1:]
        for line in handle:
            gene, counts = line.split(',', 1)
            x = np.array(counts.split(','), dtype=np.float32)
            nonzero = np.flatnonzero(x).astype(np.int32)
            genes.append(gene.strip().strip('"'))
            indices.append(nonzero)
            values.append(x[nonzero])
            indptr.append(indptr[-1]+len(nonzero))
    matrix = sparse.csr_matrix((np.concatenate(values), np.concatenate(indices), np.asarray(indptr, dtype=np.int64)),
                               shape=(len(genes), len(cells))).T.tocsr()
    meta = metadata.loc[cells]
    obs = pd.DataFrame(index=pd.Index(cells, name='cell'))
    obs['clone'], obs['lineage_block'] = _single_barcode_clones(meta.lineage_barcode)
    obs['day'] = meta.time_point.astype(int).to_numpy()
    obs['condition'] = 'day'+obs['day'].astype(str)
    obs['sample'] = meta.sample_name.astype(str).to_numpy()
    sort = obs['sample'].str.extract(r'_(high|med|low)$', expand=False)
    obs['fraction'] = sort.map(FRACTIONS).fillna('unsorted')
    obs['replicate'] = obs['sample'].str.extract(r'rep(\d)', expand=False).fillna('0')
    adata = ad.AnnData(matrix, obs=obs, var=pd.DataFrame(index=pd.Index(genes, name='gene')))
    adata.var_names_make_unique()
    adata.uns['source'] = dict(study='Watermelon (Oren et al., Nature 2021)', accession='GSE150949')
    folder.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(cache, compression='gzip')
    return adata


def holdout(adata, clone_key: str, condition_key: str, fraction: float = .2, unit: str = 'cells', min_cells: int = 2,
            seed: int = 0) -> np.ndarray:
    """Boolean mask of held-out cells for evaluation.

    * ``unit='cells'`` (seen clones): a `fraction` of the cells of every clone with at least `min_cells` cells in each
      condition is held out, so the test clones also appear in training.
    * ``unit='clones'`` (unseen clones): all cells of a random `fraction` of the clones are held out.

    Cells without a clone label are never held out.
    """
    import pandas as pd
    rng = np.random.default_rng(seed)
    clone = adata.obs[clone_key].astype(object)
    labeled = pd.notna(clone).to_numpy() & (clone.astype(str) != '').to_numpy()
    mask = np.zeros(adata.n_obs, bool)
    if unit == 'clones':
        clones = np.array(sorted(set(clone[labeled])))
        chosen = set(rng.choice(clones, int(round(fraction*len(clones))), replace=False))
        return labeled & clone.isin(chosen).to_numpy()
    if unit != 'cells':
        raise ValueError("unit must be 'cells' or 'clones'")
    condition = adata.obs[condition_key].astype(str).to_numpy()
    frame = pd.DataFrame({'clone': clone.to_numpy(), 'condition': condition, 'i': np.arange(adata.n_obs)})[labeled]
    counts = frame.groupby(['clone', 'condition']).size().unstack(fill_value=0)
    eligible = counts.index[(counts >= min_cells).sum(1) >= 2] if counts.shape[1] >= 2 else []
    for (c, _), cells in frame[frame.clone.isin(eligible)].groupby(['clone', 'condition']):
        k = max(1, int(round(fraction*len(cells))))
        mask[rng.choice(cells.i.to_numpy(), min(k, len(cells)-1), replace=False)] = True
    return mask
