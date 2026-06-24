# 📚 Sources & References

This file lists the research, papers, documentation, and internal materials that informed the design
and implementation of the models in [`main.ipynb`](main.ipynb). Sources are grouped by the part of the
project they influenced. Each model section names the techniques actually used in the notebook and the
work they come from.

> **Highlight (as requested):** the **chained MLP→SVM** models — both the **two-stage frozen-embedding**
> version and the **end-to-end (E2E) differentiable SVM head** — are covered in
> [Section 3](#3-chained-mlpsvm--two-stage-and-end-to-end-model-3). Their primary source is **Tang (2013),
> *Deep Learning using Linear Support Vector Machines*** and our internal literature review.

---

## 1. Internal project research & documents

| Document | What it provided |
|---|---|
| [`docs/research_materials/svm_architecture_research.md`](docs/research_materials/svm_architecture_research.md) | **The core literature review** (AI-assisted search via [Consensus](https://consensus.app), 50 papers screened, 26 cited). Drove every SVM and MLP→SVM design decision: linear vs. kernel SVMs, Nyström kernel approximation, `class_weight` for imbalance, `decision_function` for AUC (no calibration), two-stage vs. end-to-end chaining. Its full bibliography is reproduced in [Section 8](#8-full-bibliography-from-the-literature-review). |
| [`docs/assignment/Final_Project_Instructions.md`](docs/assignment/Final_Project_Instructions.md) | Assignment brief: task, AUC metric, 3-of-4 advanced-model requirement, runtime limit, submission format. |
| [`docs/assignment/regional_logistics_feature_descriptions.md`](docs/assignment/regional_logistics_feature_descriptions.md) | Field-level data dictionary used to drive feature engineering in Part 1–2. |
| [`docs/course_materials/StatisticsTheory_Materials.md`](docs/course_materials/StatisticsTheory_Materials.md) | Course statistics/ML theory reference. |

---

## 2. Multi-Layer Perceptron / ANN (Model 1)

Techniques used: feed-forward MLP (`Linear → BatchNorm → ReLU → Dropout`), `BCEWithLogitsLoss` with
`pos_weight` for the ~14 % imbalance, `AdamW`, `ReduceLROnPlateau`, early stopping.

- Gorishniy, Rubachev, Khrulkov, Babenko — **Revisiting Deep Learning Models for Tabular Data**, NeurIPS 2021. <https://arxiv.org/abs/2106.11959> *(tabular-MLP baseline design)*
- Srivastava, Hinton, Krizhevsky, Sutskever, Salakhutdinov — **Dropout: A Simple Way to Prevent Neural Networks from Overfitting**, JMLR 2014. <https://jmlr.org/papers/v15/srivastava14a.html>
- Ioffe & Szegedy — **Batch Normalization**, ICML 2015. <https://arxiv.org/abs/1502.03167>
- Kingma & Ba — **Adam: A Method for Stochastic Optimization**, ICLR 2015. <https://arxiv.org/abs/1412.6980>
- Loshchilov & Hutter — **Decoupled Weight Decay Regularization (AdamW)**, ICLR 2019. <https://arxiv.org/abs/1711.05101>
- Borisov et al. — **Deep Neural Networks and Tabular Data: A Survey**, IEEE TNNLS 2021. <https://doi.org/10.1109/tnnls.2022.3229161>

---

## 3. Chained MLP→SVM — two-stage and end-to-end (Model 3)

This section covers **both** chained variants implemented in the notebook.

### 3a. Two-stage (frozen-embedding) MLP→SVM — the adopted approach
Train the MLP to convergence, freeze it, extract the penultimate-layer embeddings (optionally
L2-normalized), then fit a linear/Nyström SVM on those embeddings with leakage-safe per-fold refitting.

- **Internal review §3.5 / §4** — recommends the two-stage frozen-embedding pipeline as the rigorous,
  leakage-safe default on tabular data: [`svm_architecture_research.md`](docs/research_materials/svm_architecture_research.md).
- Tang, Y. — **Deep Learning using Linear Support Vector Machines**, ICML 2013 Workshop. <https://arxiv.org/abs/1306.0239> *(refs [11][12] in the review)*
- Díaz-Vico, Prada, Omari, Dorronsoro — **Deep Support Vector Neural Networks**, Integr. Comput.-Aided Eng. 2020. <https://doi.org/10.3233/ica-200635> *(ref [13])*

### 3b. End-to-end (E2E) differentiable SVM head — the documented experiment
Replace the MLP's BCE head with a **linear-SVM (squared-hinge / L2-SVM) loss** and back-propagate
through the whole network; `weight_decay` plays the role of the SVM's `1/C`.

- **Tang, Y. — Deep Learning using Linear Support Vector Machines**, ICML 2013 Workshop.
  <https://arxiv.org/abs/1306.0239> — **the primary source for the E2E squared-hinge head** (the loss
  formulation `mean(w · max(0, 1 − y·f(x))²)` and the {−1,+1} target mapping used in the notebook).
- Internal review §3.5, §4 and the **Claims & Evidence** table — conclude E2E deep+SVM heads give only
  marginal, unproven gains on tabular data, which our experiment confirmed (kept-but-not-adopted):
  [`svm_architecture_research.md`](docs/research_materials/svm_architecture_research.md).
- Borisov et al. 2021 (above) and Shmuel et al. 2025 (below) — evidence that tree/MLP baselines
  dominate deep+SVM on structured data.

---

## 4. Standalone SVM (Model 2)

Techniques used: `LinearSVC` (LIBLINEAR), RBF approximation via `Nystroem` + linear SVM, a full-RBF
`SVC` on a stratified subsample, `class_weight="balanced"`, scoring by `decision_function` (no calibration).

- Cortes & Vapnik — **Support-Vector Networks**, Machine Learning 1995. <https://doi.org/10.1007/BF00994018> *(foundational SVM)*
- Fan, Chang, Hsieh, Wang, Lin — **LIBLINEAR: A Library for Large Linear Classification**, JMLR 2008. <https://doi.org/10.5555/1390681.1442794> *(ref [1]; `LinearSVC` backend)*
- Williams & Seeger — **Using the Nyström Method to Speed Up Kernel Machines**, NeurIPS 2001. <https://proceedings.neurips.cc/paper/2000/hash/19de10adbaa1b2ee13f77f679fa1483a-Abstract.html> *(Nyström approximation)*
- Rahimi & Recht — **Random Features for Large-Scale Kernel Machines (RFF)**, NeurIPS 2007. <https://papers.nips.cc/paper_files/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html>
- Liu, Huang, Chen, Suykens — **Random Features for Kernel Approximation: A Survey**, IEEE TPAMI 2020. <https://doi.org/10.1109/tpami.2021.3097011> *(ref [6])*
- Cano, Pascual, Dorronsoro — **Nyström and RFF Ensembles for Large-Scale Kernel Predictions**, 2024. <https://doi.org/10.1007/978-3-031-74183-8_14> *(ref [5])*
- Akram-Ali-Hammouri et al. — **Fast Support Vector Classification for Large-Scale Problems**, IEEE TPAMI 2021. <https://doi.org/10.1109/tpami.2021.3085969> *(ref [20])*
- **Imbalance handling:** Rosales-Pérez, García, Herrera 2022 <https://doi.org/10.1109/tcyb.2022.3163974> [3]; Abdelhamid & Desai 2024 <https://doi.org/10.48550/arxiv.2409.19751> [22]; Hazarika & Gupta 2020 <https://doi.org/10.1007/s00521-020-05240-8> [24].
- **Calibration vs. AUC (why we skip calibration):** Syafi'ah et al. 2025 <https://doi.org/10.18860/jrmm.v4i6.33438> [4]; Böken 2020 <https://doi.org/10.1016/j.is.2020.101641> [10]; Platt — *Probabilistic Outputs for Support Vector Machines*, 1999.

---

## 5. Decision Tree (Model 5)

Techniques used: `DecisionTreeClassifier` with `class_weight="balanced"`, complexity grid over
`max_depth` / `min_samples_leaf` / **cost-complexity pruning** (`ccp_alpha`), Youden-J threshold.

- Breiman, Friedman, Olshen, Stone — **Classification and Regression Trees (CART)**, Wadsworth, 1984. *(tree induction + cost-complexity pruning)*
- scikit-learn — **Decision Trees user guide** (incl. minimal cost-complexity pruning). <https://scikit-learn.org/stable/modules/tree.html>

---

## 6. Logistic Regression (Model 4)

- scikit-learn — **`LogisticRegression`** (lbfgs solver, `class_weight="balanced"`, `C` tuned by CV).
  <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>

---

## 7. Preprocessing & evaluation techniques

- **Target (mean) encoding:** Micci-Barreca — *A Preprocessing Scheme for High-Cardinality Categorical
  Attributes*, SIGKDD Explorations 2001. <https://doi.org/10.1145/507533.507538> · scikit-learn
  [`TargetEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html)
  (leakage-safe internal cross-fitting).
- **KNN imputation:** scikit-learn [`KNNImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html) (subclassed to a median aggregation in the notebook).
- **Threshold selection — Youden's J:** Youden — *Index for Rating Diagnostic Tests*, Cancer 1950. <https://doi.org/10.1002/1097-0142(1950)3:1%3C32::AID-CNCR2820030106%3E3.0.CO;2-3>
- **Cross-validation / ROC-AUC:** scikit-learn [`StratifiedKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html), [`roc_auc_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html), [`roc_curve`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html).
- **Structured-data benchmark context:** Shmuel, Glickman, Lazebnik — *A Comprehensive Benchmark of ML
  and DL Models on Structured Data*, Neurocomputing 2025. <https://doi.org/10.1016/j.neucom.2025.131337> [19].

---

## 8. Libraries & tooling

- **scikit-learn** — Pedregosa et al., JMLR 2011. <https://jmlr.org/papers/v12/pedregosa11a.html> · docs <https://scikit-learn.org/stable/>
- **PyTorch** — Paszke et al., NeurIPS 2019. <https://arxiv.org/abs/1912.01703> · docs <https://pytorch.org/docs/stable/>
- **NumPy** — Harris et al., Nature 2020. <https://doi.org/10.1038/s41586-020-2649-2>
- **pandas** — McKinney, SciPy 2010. <https://doi.org/10.25080/Majora-92bf1922-00a> · docs <https://pandas.pydata.org/docs/>
- **Matplotlib** — Hunter, CiSE 2007. <https://doi.org/10.1109/MCSE.2007.55>
- **seaborn** — Waskom, JOSS 2021. <https://doi.org/10.21105/joss.03021>

Key API references: [`BCEWithLogitsLoss`](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) ·
[`AdamW`](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) ·
[`ReduceLROnPlateau`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html) ·
[`LinearSVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) ·
[`SVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) ·
[`Nystroem`](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.Nystroem.html).

---

## 9. Full bibliography from the literature review

The 26 references cited in [`docs/research_materials/svm_architecture_research.md`](docs/research_materials/svm_architecture_research.md), with resolvable links:

1. Fan R., Chang K., Hsieh C., Wang X., Lin C. **LIBLINEAR: A Library for Large Linear Classification.** *JMLR* 2008;9:1871–1874. <https://doi.org/10.5555/1390681.1442794>
2. Yang Y., Mirzaei G. **Performance analysis of data resampling on class imbalance…** *PLOS ONE* 2024;19. <https://doi.org/10.1371/journal.pone.0293607>
3. Rosales-Pérez A., García S., Herrera F. **Handling Imbalanced Classification Problems With SVMs via Evolutionary Bilevel Optimization.** *IEEE Trans. Cybernetics* 2022;53:4735–4747. <https://doi.org/10.1109/tcyb.2022.3163974>
4. Syafi'ah N., Jamhuri M., Pranata F.I., et al. **Cross-Dataset Evaluation of SVMs: A Reproducible, Calibration-Aware Baseline for Tabular Classification.** *J. Riset Mahasiswa Matematika* 2025. <https://doi.org/10.18860/jrmm.v4i6.33438>
5. Cano B., Pascual Á.F., Dorronsoro J.R. **Nyström and RFF Ensembles for Large-Scale Kernel Predictions.** 2024. <https://doi.org/10.1007/978-3-031-74183-8_14>
6. Liu F., Huang X., Chen Y., Suykens J. **Random Features for Kernel Approximation: A Survey.** *IEEE TPAMI* 2020;44:7128–7148. <https://doi.org/10.1109/tpami.2021.3097011>
7. Nguyen M., Vien N.A. **Scalable and Interpretable One-class SVMs with Deep Learning and Random Fourier features.** 2018. <https://doi.org/10.1007/978-3-030-10925-7_10>
8. Odesola P.A., Adegoke A., Babalola I. **Model uncertainty quantification: A post hoc calibration approach…** *J. Eng. Res. Sci.* 2025. <https://doi.org/10.1101/2025.09.28.25336834>
9. Manokhin V., Gronhaug D. **Classifier Calibration at Scale.** *ArXiv* 2026; abs/2601.19944. <https://doi.org/10.48550/arxiv.2601.19944>
10. Böken B. **On the appropriateness of Platt scaling in classifier calibration.** *Inf. Syst.* 2020;95:101641. <https://doi.org/10.1016/j.is.2020.101641>
11. Tang Y. **Deep Learning using Linear Support Vector Machines.** *arXiv: Learning* 2013. <https://arxiv.org/abs/1306.0239>
12. Tang Y. **Deep Learning using Support Vector Machines.** *ArXiv* 2013; abs/1306.0239. <https://arxiv.org/abs/1306.0239>
13. Díaz-Vico D., Prada J., Omari A., Dorronsoro J.R. **Deep support vector neural networks.** *Integr. Comput.-Aided Eng.* 2020;27:389–402. <https://doi.org/10.3233/ica-200635>
14. Yang X., Meng P., Jiang Z., Zhou L. **Deep siamese residual SVM with applications to disease prediction.** *Comput. Biol. Med.* 2025;196:110693. <https://doi.org/10.1016/j.compbiomed.2025.110693>
15. Li Y., Zhang T. **Deep neural mapping support vector machines.** *Neural Networks* 2017;93:185–194. <https://doi.org/10.1016/j.neunet.2017.05.010>
16. Moutaouakil K.E., Roudani M., Ouhmid A., Zhilenkov A., Mobayen S. **Decomposition and Symmetric Kernel Deep Neural Network Fuzzy SVM.** *Symmetry* 2024;16:1585. <https://doi.org/10.3390/sym16121585>
17. Wu Y., Li H., Chen Y., Lee W.W. **Intrusion Detection Model Based on Multi-Kernel Approximation and DNN.** *Security and Privacy* 2025;8. <https://doi.org/10.1002/spy2.70117>
18. Borisov V., Leemann T., Sessler K., Haug J., Pawelczyk M., Kasneci G. **Deep Neural Networks and Tabular Data: A Survey.** *IEEE TNNLS* 2021;35:7499–7519. <https://doi.org/10.1109/tnnls.2022.3229161>
19. Shmuel A., Glickman O., Lazebnik T. **A comprehensive benchmark of ML and DL models on structured data.** *Neurocomputing* 2025;655:131337. <https://doi.org/10.1016/j.neucom.2025.131337>
20. Akram-Ali-Hammouri Z., Fernández-Delgado M., Cernadas E., Barro S. **Fast Support Vector Classification for Large-Scale Problems.** *IEEE TPAMI* 2021;44:6184–6195. <https://doi.org/10.1109/tpami.2021.3085969>
21. Muthukumar V., Narang A., Subramanian V., Belkin M., Hsu D.J., Sahai A. **Classification vs regression in overparameterized regimes.** *JMLR* 2020;22:222. <https://www.jmlr.org/papers/v22/20-603.html>
22. Abdelhamid M., Desai A. **Balancing the Scales: Tackling Class Imbalance in Binary Classification.** *ArXiv* 2024; abs/2409.19751. <https://doi.org/10.48550/arxiv.2409.19751>
23. Purwadi J., Fithriasari K., Kuswanto H. **An Adaptive Robust SVM With SMO for Imbalanced Data Classification.** *IEEE Access* 2026;14:17031–17038. <https://doi.org/10.1109/access.2026.3655329>
24. Hazarika B.B., Gupta D. **Density-weighted support vector machines for binary class imbalance learning.** *Neural Comput. Appl.* 2020;33:4243–4261. <https://doi.org/10.1007/s00521-020-05240-8>
25. Wu Y., Liu Y. **Robust Truncated Hinge Loss Support Vector Machines.** *JASA* 2007;102:974–983. <https://doi.org/10.1198/016214507000000617>
26. Choudhary R., Shukla S. **Reduced-Kernel Weighted Extreme Learning Machine Using Universum Data (RKWELM-UFS).** *Symmetry* 2022;14:379. <https://doi.org/10.3390/sym14020379>

---

<sub>Compiled for the Intro to Machine Learning final project (Group #15). The literature review in
`docs/research_materials/` was produced with [Consensus](https://consensus.app); library documentation
links point to the official project docs.</sub>
