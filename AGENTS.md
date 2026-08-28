# AGENTS.md

## Research purpose

OmniXAS predicts site-specific X-ray absorption near-edge spectra (XANES) from material structures. It is a fast surrogate for expensive FEFF and VASP spectral calculations, aimed at high-throughput and real-time XAS analysis.

Primary reference: [OmniXAS: A Universal Deep-Learning Framework for Materials X-ray Absorption Spectra](https://arxiv.org/abs/2409.19552).

## Core scientific pipeline

```text
Pymatgen crystal structure
→ periodic material graph
→ M3GNet structural encoder
→ 64D latent vector at the absorbing atom
→ XASBlock multilayer perceptron
→ 141-point nonnegative XANES spectrum
```

OmniXAS does not use M3GNet's energy prediction directly. It transfers M3GNet's learned structural representation to the XAS task.

### M3GNet encoder

- Model: `M3GNet-MP-2021.2.8-PES`, stored locally under `models/`.
- Implementation: MatGL 0.8.5 with PyTorch and DGL.
- Original task: universal interatomic potential trained on Materials Project relaxation data across 89 elements.
- Representation: atoms are nodes, neighbor interactions are edges, and line-graph operations encode bond-angle/three-body interactions.
- Standard OmniXAS feature: the final 64D node state after three M3GNet blocks, selected at the absorbing-site index.
- Main implementation: `omnixas/featurizer/m3gnet_featurizer.py`.

The pretrained model is kept locally because loading another model revision can silently change the latent features and invalidate trained XAS heads.

### XAS prediction head

`omnixas/model/xasblock.py` defines the XASBlock:

- hidden layers: `Linear → BatchNorm1d → SiLU → Dropout`;
- output layer: `Linear → Softplus`;
- Softplus enforces nonnegative spectral intensities;
- current target dimension is 141 energy-grid values;
- dropout is the class variable `XASBlock.DROPOUT`, so set it before constructing a model.

Training uses Adam, mean squared error, validation monitoring, checkpointing, and optional learning-rate schedulers through `XASBlockRegressor`.

## Data and physical scope

The paper dataset contains 75,691 curated, site-specific K-edge spectra for eight 3d transition metals:

`Ti, V, Cr, Mn, Fe, Co, Ni, Cu`

Two simulation fidelities are used:

- **FEFF:** real-space multiple-scattering spectra for all eight elements.
- **VASP:** higher-cost core-hole calculations for Ti and Cu.

The ML-ready spectra are cleaned, broadened, edge-aligned, interpolated to a shared 141-point grid, and scaled. They are not raw simulator outputs. Existing tutorial feature and target files use a factor-of-1000 scale; preserve the established scale unless an experiment explicitly changes it.

Important locations:

- `tutorial_omnixas/ml_data/`: aligned `X` features and `y` spectra.
- `tutorial_omnixas/material_id_and_site/`: row-aligned material/site identifiers.
- `tutorial_omnixas/FEFF/`: local Cu structure/simulation examples.
- `models/M3GNet-MP-2021.2.8-PES/`: pinned encoder model.

For every split, ID rows, feature rows, and target rows must remain in exactly the same order.

## Model families and transfer learning

Use the paper terminology consistently:

- **ExpertXAS:** trained from scratch for one element and one simulation fidelity.
- **UniversalXAS:** one head trained on combined FEFF data from all eight elements.
- **Tuned-UniversalXAS:** UniversalXAS fine-tuned on one element, optionally at VASP fidelity.

The research tests three transfer mechanisms:

1. **Feature transfer:** pretrained M3GNet latent states replace hand-designed structural descriptors.
2. **Hierarchical transfer:** all-element UniversalXAS knowledge is adapted to an element-specific task.
3. **Cross-fidelity transfer:** a FEFF foundation model is adapted to higher-cost VASP spectra.

## Evaluation

The main paper metric is eta:

```text
eta = median MSE of the train-mean-spectrum baseline
      ------------------------------------------------
      median per-spectrum MSE of the model
```

Higher eta is better. Build the baseline from the training targets of the same element/fidelity, then evaluate it on the requested validation or test targets.

Also report aggregate MSE and median per-spectrum MSE when useful. Do not compare eta values computed with different baselines or preprocessing.

### Scientific invariants

- Split by material, not by absorbing site. All sites from one material belong to one split.
- Never select models, checkpoints, features, or hyperparameters using test eta.
- Select using validation metrics, then evaluate the test set once.
- Never mix a head trained on one encoder's feature space with features from another encoder run.
- Preserve the published split membership when comparing against paper values.
- Verify feature dimensions, target dimensions, row counts, and ID ordering before training.
- Record random seed, source checkpoint, feature provenance, preprocessing scale, and hyperparameters for every run.
- Missing data or checkpoints must fail clearly; do not substitute another run silently.

## Notebook roles

- `tutorial_omnixas/OmniXAS_tutorial.ipynb`: introductory Cu workflow.
- `tutorial_omnixas/reproduce_paper_training.ipynb`: best reference for paper-style training and evaluation.
- `tutorial_omnixas/compare_ti_expert_universal.ipynb`: focused expert/universal/fine-tuned comparison.
- `tutorial_omnixas/cu_feff_encoder_study.ipynb`: experimental supervised encoders and local pooling.
- `tutorial_omnixas/analyze_all8_feff_training071326.ipynb`: analysis of external all-element encoder runs.
- `tutorial_omnixas/finetune_all8_feff_elements.ipynb`: element adaptation of custom UniversalXAS heads.
- `tutorial_omnixas/m3gnet_last_block_finetune_omnixas.ipynb`: experimental M3GNet block fine-tuning.

Treat `.ipynb_checkpoints/` as stale editor state, not maintained source.

## Known hazards

- `omnixas/data/material_split.py` has a suspicious greedy assignment using `diffs.index(max(diffs))`. Do not trust newly generated material splits with uneven site counts until this is fixed and tested. Published text-file splits are unaffected.
- `tutorial_omnixas/material_id_and_site/sanitycheck2.ipynb` is broken and contains a hard-coded Materials Project credential. Never copy or reuse embedded credentials; use `MP_API_KEY` from the environment and rotate exposed keys.
- Several research notebooks require ignored `output/` checkpoints, external raw data, or `../training071326`. A clean clone may not contain these artifacts.
- Some experimental notebooks monkey-patch private MatGL internals for GPU compatibility. Keep such patches local to experiments and pinned to MatGL 0.8.5.
- Do not execute notebooks blindly: some enable long GPU training or overwrite checkpoint directories by default.
- Only pass fields supported by `XASBlockRegressorConfig`; unknown Pydantic fields may be ignored and create misleading experiment settings.

## Environment

- Required Python version: 3.11.
- Key pinned packages: PyTorch 2.1.0, torchdata 0.7.0, MatGL 0.8.5, DGL, Pymatgen, and Lightning.
- PyTorch is pinned because the required DGL stack is not supported by newer project configurations.

## Change workflow

1. Identify whether the change affects preprocessing, feature extraction, head training, or evaluation.
2. Preserve the scientific invariants above.
3. Prefer existing paper splits and model definitions over introducing new variants.
4. Run a small CPU/static or smoke check before any full GPU experiment.
5. For non-trivial data or metric logic, add a focused runnable test.
6. Do not commit generated checkpoints, raw data, Lightning logs, notebook progress output, API keys, or temporary experiment files.
7. State clearly when a result is exploratory rather than directly comparable to the paper.
