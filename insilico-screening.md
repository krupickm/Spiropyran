# In Silico Combinatorial Screening for Spiropyran Photoswitches

**Project scope:** Build a computational pipeline that enumerates synthetically accessible spiropyran precursors from commercial building blocks, computes molecular descriptors at multiple levels of theory, and uses ML to predict diastereomeric ratios and liquid crystal behaviour.

**Context:** This supports the GACR grant WP4 (computational screening). The goal is to move from "we'll use ML" (vague) to a concrete, working pipeline.

---

## 1. The Problem

Spiropyran ring closure produces two diastereomers (anti/syn) at the spiro centre. The diastereomeric ratio (d.r.) is determined by kinetic control at the transition state / conical intersection, not by ground-state thermodynamics. Existing d.r. values are modest (up to ~79:21, i.e., ?G ˜ 3 kJ/mol) — the energy landscape is flat. We need to screen large numbers of candidate structures computationally to find substitution patterns that improve d.r., without synthesising everything.

A secondary target is predicting liquid crystal (LC) behaviour of the resulting spiropyrans — here, we genuinely don't know which molecular features govern good LC properties, so we take a descriptor shotgun approach.

---

## 2. Pipeline Overview

```
Sigma catalogue (SDF)
        ¦
        ?
  Substructure filter --? arylhydrazines + ketones
        ¦
        ?
  Fischer indole enumeration (SMARTS reaction template)
        ¦
        ?
  Product filtering (property + substructure exclusion)
        ¦
        ?
  3D conformer generation
        ¦
        ?
  Descriptor calculation (chemoinformatic + geometric + electronic)
        ¦
        ?
  ML prediction (d.r. and/or LC properties)
```

---

## 3. Step 1: Building Block Sourcing

**Source:** Sigma-Aldrich downloadable SDF catalogue (also available via eMolecules or ZINC aggregators).

**Filtering:** Substructure search in RDKit for:
- Arylhydrazines (reactant A)
- Ketones — preferably symmetrical or methyl ketones to avoid Fischer regiochemistry issues (reactant B)

**Note on regiochemistry:** Unsymmetrical ketones give two possible indoles via Fischer synthesis. Options: restrict to symmetrical/methyl ketones, or enumerate both regioisomers and keep both for screening.

---

## 4. Step 2: Reaction Enumeration

**Tool:** RDKit `AllChem.ReactionFromSmarts()`

Encode the Fischer indole synthesis as a SMARTS reaction template. Apply it to the Cartesian product of arylhydrazines × ketones. Output: library of 3H-indole product SMILES.

This is standard reaction-based virtual library enumeration — a well-established chemoinformatics technique used routinely in pharma for combinatorial library design.

---

## 5. Step 3: Product Filtering

Two layers of filters applied to the enumerated library:

### 5a. Property filters (numerical cutoffs)
- Molecular weight cap
- Rotatable bonds = 4–5 (main proxy for "weird long chains")
- Ring count within reasonable range
- H-bond donor/acceptor count (proxy for functional group load)

### 5b. Substructure exclusion filters (SMARTS-based)
Reject molecules containing problematic groups that won't survive spiropyran synthesis conditions (acid, aldehyde condensation, light). Examples:
- Nitro groups (reduction side reactions)
- Free thiols
- Acid chlorides (too reactive)
- Heavy halogens if unwanted
- Long linear alkyl chains (=6 carbons)

### 5c. Substructure inclusion filters
- Must have free NH on indole (needed for downstream chemistry)
- Must have substitution pattern compatible with spiropyran formation

### 5d. Synthetic Accessibility Score (SA Score)
Ertl & Schuffenhauer (2009), available in RDKit. Scale 1 (easy) to 10 (hard). Based on fragment frequency in PubChem + complexity penalty. Useful for ranking/triage, but not a substitute for domain-specific filters — SA Score doesn't know about spiropyran chemistry.

Alternative: SCScore (Coley et al., 2018) — estimates synthetic steps from commercial starting materials using a neural network trained on reaction databases. More principled but less interpretable.

**Key insight:** For this pipeline, generic SA scores are secondary. The real value is in bespoke substructure filters encoding domain knowledge about what makes a good spiropyran precursor.

---

## 6. Step 4: Descriptor Calculation

Three levels of descriptors, increasing in computational cost:

### 6a. Chemoinformatic descriptors (RDKit + Mordred)

**RDKit** core functions:
- Molecular I/O (SMILES ? Mol objects, SDF reading/writing)
- Substructure matching (SMARTS pattern search)
- Reaction handling (SMARTS-encoded transformations)
- Descriptor calculation (MW, logP, rotatable bonds, TPSA, ring counts, etc.)
- Fingerprints (Morgan/ECFP, MACCS — for similarity search and ML input)
- Stereochemistry handling (chiral centre detection, R/S assignment, stereoisomer enumeration)
- Conformer generation (3D embedding + force field optimisation)
- Fragmentation (BRICS — retrosynthetic decomposition into building blocks)
- MCS (maximum common substructure — for clustering library into scaffold families)
- Murcko scaffolds (strip side chains to core ring system)
- Constrained embedding (fix part of the molecule to known coordinates — useful for MECP scaffold approach)

**Mordred** — exhaustive descriptor calculator, ~1800 descriptors per molecule:
- Constitutional (atom/bond counts, MW, composition)
- Topological (Wiener index, Zagreb indices, Kier-Hall chi — branching/shape from graph)
- Geometric / 3D (moments of inertia, WHIM, GETAWAY — requires conformer)
- Electronic / charge (Gasteiger charges, TPSA — approximate)
- Fragment-based (functional group counts, E-state indices)
- Information-content (Shannon entropy of molecular graph — complexity/symmetry)

**Workflow:** Feed Mol objects ? get pandas DataFrame (molecules × descriptors) ? clean NaNs ? feed to ML.

### 6b. Steric descriptors (morfeus)

**morfeus** — purpose-built for 3D steric descriptors from coordinates:
- Buried volume
- Cone angles
- Sterimol parameters (L, B1, B5)

Directly relevant for quantifying steric bulk at the 3' position in each diastereomer. Answers the question: "how much steric hindrance does this substituent actually present?"

### 6c. Electronic descriptors (DFT + Multiwfn)

From a DFT wavefunction (Gaussian .fchk, ORCA output, or .wfn/.wfx files):
- Orbital energies (HOMO, LUMO, gap)
- Dipole and quadrupole moments
- Polarisability tensor and its anisotropy (key for LC behaviour)
- Partial charges (Mulliken, NBO, CHELPG, Hirshfeld)
- Electrostatic potential surface statistics (min, max, variance on isodensity surface)
- Fukui indices (local reactivity)
- Bond orders, electron density at bond critical points (QTAIM/AIM analysis)

**Multiwfn** — the main extraction tool. Reads .fchk / .wfn / .wfx files, computes all of the above, dumps as text tables.

**cclib** — Python library that parses QC output files and returns orbital energies, charges, dipoles as Python objects. Bridge back to pandas/scikit-learn.

**ASE (Atomic Simulation Environment)** — Python wrapper for automating QC calculations across hundreds of molecules (input generation, job management, output parsing).

**Workflow:** RDKit conformers ? xTB/DFT calculations ? Multiwfn/cclib extraction ? concatenate with Mordred descriptor matrix ? ML.

---

## 7. Fragment-Aware Descriptors

Instead of one global descriptor vector per molecule, decompose into local environments:

### 7a. DScribe
Computes atom-environment descriptors:
- SOAP (Smooth Overlap of Atomic Positions)
- ACSF (Atom-Centred Symmetry Functions — related to Behler)
- MBTR (Many-Body Tensor Representation)

Each atom gets a vector describing its local geometric and chemical environment. Can compute descriptors specifically for the spiro centre environment, the chromene ring, or the indoline nitrogen.

### 7b. QML (von Lilienfeld group)
Molecular representations derived from electronic structure:
- Coulomb matrix
- SLATM
- FCHL

Not "compute all descriptors" but "represent the molecule so ML captures everything."

### 7c. The key idea for spiropyrans
The d.r. is determined locally at the spiro centre, not by global molecular properties. So: compute DScribe SOAP descriptors around the spiro carbon for both diastereomers ? feed into ML ? predict d.r. This is physically motivated.

---

## 8. ML Strategy

### 8a. For diastereomeric ratio prediction
- ~40 experimental data points (small data regime)
- Possibly hundreds from computation
- Target: one number (d.r.) per molecule
- **Gaussian processes** preferred — give uncertainty estimates, work well with small data, enable active learning (compute next molecule where model is most uncertain)
- Random forest as baseline (feature importance for free)
- Behler-Parrinello NN potentials are overkill (they learn full PES from thousands of DFT points per molecule — we only need energy at two specific geometries)

### 8b. For liquid crystal property prediction
- QSPR (Quantitative Structure-Property Relationships) approach
- Compute full descriptor matrix (Mordred 1800 + DFT electronic + morfeus steric)
- Pair with experimental LC transition temperatures or phase classification
- Feature selection: random forest importance, LASSO, mutual information
- Literature consistently finds length-to-breadth ratio and polarisability anisotropy as top features (physically sensible)
- Random forest / XGBoost as workhorse models
- **Important:** compute descriptors for BOTH SP and MC forms, since photoswitching toggles between them and LC behaviour depends on which form is present

### 8c. Active learning loop
Gaussian process predicts d.r. with uncertainty ? pick the molecule with highest uncertainty ? compute it (xTB/DFT) ? add to training set ? retrain ? repeat. This is the efficient way to grow the training set without brute-force computing everything.

---

## 9. Software Stack Summary

| Layer | Tool | Role |
|-------|------|------|
| Building blocks | Sigma SDF, eMolecules, ZINC | Commercial catalogue |
| Enumeration | RDKit | SMARTS reaction, Cartesian product |
| Filtering | RDKit | Property + substructure filters, SA Score |
| Chemoinformatic descriptors | Mordred (on RDKit) | ~1800 descriptors per molecule |
| 3D conformers | RDKit | ETKDG embedding + MMFF optimisation |
| Steric descriptors | morfeus | Buried volume, Sterimol, cone angles |
| Semi-empirical | GFN2-xTB | Fast geometry opt, approximate energies |
| DFT | ORCA / Gaussian | Wavefunction for electronic descriptors |
| Wavefunction analysis | Multiwfn | ESP, charges, polarisability, QTAIM |
| QC output parsing | cclib | QC output ? Python objects |
| Automation | ASE | Job management for batch QC calculations |
| Local environment descriptors | DScribe | SOAP / ACSF around spiro centre |
| ML | scikit-learn, GPy | Gaussian processes, random forest, XGBoost |

---

## 10. Key References to Look Up

- Ertl & Schuffenhauer (2009) — SA Score
- Coley et al. (2018) — SCScore
- Behler & Parrinello (2007) — atom-centred symmetry functions (conceptual basis for DScribe)
- von Lilienfeld group — QML library, Coulomb matrix, FCHL representations
- Mordred documentation — descriptor definitions
- DScribe documentation — SOAP, ACSF implementations
- RDKit Cookbook — reaction enumeration examples

---

## 11. Open Questions / Next Steps

1. Get Sigma SDF catalogue downloaded and filtered for arylhydrazines + ketones
2. Write the Fischer indole SMARTS template (handle regiochemistry decision)
3. Define the substructure exclusion list specific to spiropyran synthesis compatibility
4. Decide descriptor scope for first pass (Mordred only? + morfeus? + DFT?)
5. Set up active learning loop with Gaussian processes once first d.r. predictions exist
6. For LC prediction: find published QSPR datasets for LC transition temperatures to benchmark against before applying to spiropyrans