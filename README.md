# allograph-clustering

An experiment in allograph induction: given every image of one character cut from historical manuscripts
and printed books, find the distinct ways it is written, name them, and label each image with one.
The first character is 候 (U+5019) in Japanese sources, whose abbreviations have no code point of
their own.

## Terms

| Term | Meaning here |
|---|---|
| Character | The unit being written, named by its code point: 候 is U+5019. |
| Allograph | A recurring way of writing a character. 候 written out and its ん-like abbreviation are two allographs of 候. |
| Crop | One image of the character, cut from a page. |
| Allograph induction | Deriving a character's allographs from unlabelled crops: clustering, then a person naming categories. |
| Clustering | Grouping crops by stroke shape without labels. It yields shape clusters (L1 to L24), numbered per run. |
| Classifier cluster | A cluster of the character classifier's embedding, C1 upwards. Each pack carries them, for comparison. |
| Category | An allograph as a person named it, made of one or more shape clusters. |
| Classification | Assigning every crop to one category. |
| Label | The category recorded for a crop. Labelling writes one label per crop. |

## 候

12,453 crops in the pack (classifier clustering revision `582865e3b2b68c53`), 11,990 of them labelled by run
`191a10b35330`; the rest had too little contrast to restore.

| Category | Crops | Shape clusters | Drawing |
|---|---:|---|---|
| 候, written out, from full to cursive | 2,640 | L1–L5 | |
| ん / h shape | 724 | L6–L7 | <img src="categories/drawings/n-shape.svg" width="48" alt="ん-like abbreviation of 候"> |
| Zigzag / ゝ | 2,504 | L8–L12 | <img src="categories/drawings/dotted.svg" width="48" alt="Kinked stroke with a blot at the turn"> <img src="categories/drawings/long-ten.svg" width="48" alt="Tall kinked stroke"> |
| Contained stroke | 6,122 | L13–L24 | <img src="categories/drawings/contained-dot.svg" width="48" alt="Single short stroke"> |

The categories are in [`categories/U+5019.yaml`](categories/U+5019.yaml) and every crop's label in
[`labels/U+5019.tsv`](labels/U+5019.tsv). The drawings are editorial normalisations, one filled path in
a 1000-unit em, and are not attested glyphs.

### Open questions

- No crop has a label assigned by a person, so no clustering or classification here has a measured
  accuracy. A hand-labelled sample is the first thing a comparison of methods needs.
- "Contained stroke" holds half the crops and twelve shape clusters. How much of it is one allograph, and
  how much is fragments or badly cut crops, is unknown.
- The four categories follow the top branches of the shape tree. They have not been checked against the
  forms of 候 that kuzushiji dictionaries list.
- Shape cluster numbers belong to one training run, so the categories file has to be rewritten after
  training again.

## Run

```sh
uv sync --all-extras --group dev
gh release download packs -p 'U+5019.npz' -D packs
uv run allograph-clustering cluster U+5019
uv run allograph-clustering label U+5019
uv run allograph-clustering desk U+5019
```

`cluster` trains a restoration network and two shape encoders on the character's own crops and builds
the shape tree under `runs/U+5019/`. It takes 15 to 30 minutes for 12,000 crops on a CUDA GPU and needs at
least 500 usable crops. `label` classifies every crop into the categories of `categories/U+5019.yaml` and
writes `labels.tsv` and `report.html` into the run. With no categories file it writes
`runs/U+5019/proposal.yaml`, six unnamed categories from the top branches of the tree, to be edited,
named and moved to `categories/`.

`desk` serves the same report at http://127.0.0.1:8760/ and classifies again whenever the categories file
changes: edit the file, reload the page.

## Method

1. Frame each crop at 96 px with its ink mask (Otsu, specks and edge fragments removed) and score its
   quality.
2. Train a U-Net on the crisp crops, degraded on purpose, and restore every crop to clean ink.
3. Train two contrastive encoders (SimCLR, ResNet-18, no pretrained weights) on the restored ink, with
   alterations of position, size, slant, ink weight and texture, so that only stroke shape is left to
   compare.
4. Reduce the features to ten dimensions with UMAP and cut a Ward tree into 24 shape clusters.
5. A person groups the shape clusters into categories and marks the one that is the character written out.
6. The classifier embedding decides written character or abbreviation; the shape clusters sort the
   abbreviations.

## Data

A pack is one `.npz` file with every crop of a character at most 160 px on the long side, its
classifier embedding and its classifier cluster; `src/allograph_clustering/pack.py` describes the fields.
Packs are attached to the [`packs` release](https://github.com/mkpoli/allograph-clustering/releases/tag/packs).

The crops, their transcriptions and the classifier clusters come from
[Glyph Atlas](https://glyphatlas.org/en/forms/U+5019) (CC BY-SA 4.0). The part of a crop's id before the
first colon names the dataset its image comes from, and each image keeps that dataset's licence:

| Id | Dataset | Licence | Crops of 候 |
|---|---|---|---:|
| `hi` | 東京大学史料編纂所「くずし字データセット」[doi:10.57459/hi.34.kuzushiji](https://doi.org/10.57459/hi.34.kuzushiji) | CC BY 4.0 | 10,813 |
| `codh` | 『日本古典籍くずし字データセット』（国文研ほか所蔵／CODH加工）[doi:10.20676/00000340](https://doi.org/10.20676/00000340) | CC BY-SA 4.0 | 1,640 |

One crop of 候 from a third dataset, whose image licence depends on the holding institution, is left out
of the pack.

## Layout

| Path | |
|---|---|
| `src/allograph_clustering/` | `pack`, `ink`, `restore`, `encoder`, `cluster`, `classify`, `report`, `desk`, `cli` |
| `categories/<code>.yaml` | the categories a person named for one character, and `drawings/` |
| `labels/<code>.tsv` | every crop's classifier cluster, shape cluster and label under those categories |
| `packs/`, `runs/` | downloaded packs and training output, not committed |

## Sources

- Chen, Kornblith, Norouzi & Hinton 2020, "A Simple Framework for Contrastive Learning of Visual
  Representations", [arXiv:2002.05709](https://arxiv.org/abs/2002.05709).
- He, Zhang, Ren & Sun 2016, "Deep Residual Learning for Image Recognition",
  [arXiv:1512.03385](https://arxiv.org/abs/1512.03385).
- McInnes, Healy & Melville 2018, "UMAP: Uniform Manifold Approximation and Projection for Dimension
  Reduction", [arXiv:1802.03426](https://arxiv.org/abs/1802.03426).
- Ronneberger, Fischer & Brox 2015, "U-Net: Convolutional Networks for Biomedical Image Segmentation",
  [arXiv:1505.04597](https://arxiv.org/abs/1505.04597).
- 金沢クレハ (刈羽マキ), Discord, 24 October 2026 (personal communication, unpublished): the suggestion to
  replace hand-written cleaning with a learned model, proposed as a variational autoencoder followed by
  k-means.

Code under the MIT licence.
