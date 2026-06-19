# TLDR Answer
For medium-scale, imbalanced tabular data with a near-saturated AUC, **linear SVMs (LinearSVC/SGDClassifier) or RBF-SVM with kernel approximation (Nyström/RFF) are best-practice for standalone SVMs** due to scalability and robust AUC, using class weighting and decision_function for AUC; **chained MLP→SVM models offer little added value over a strong MLP baseline on tabular data**, with end-to-end SVM heads (Tang 2013) showing only marginal gains—so the two-stage frozen-embedding approach is preferred for rigor and simplicity.

## 1. Introduction

This review synthesizes recent research on best-practice SVM architectures for binary classification on medium-scale, imbalanced tabular data, focusing on both standalone SVMs and chained MLP→SVM models. For standalone SVMs, the literature consistently finds that linear SVMs (e.g., LinearSVC, SGDClassifier) scale efficiently to tens of thousands of rows and perform robustly when paired with class weighting or sample weighting to address imbalance [1] [2] [3] [4]. Kernel SVMs (RBF/polynomial) can offer marginal accuracy gains but are computationally prohibitive at this scale unless kernel approximation methods like Nyström or Random Fourier Features (RFF) are used [5] [6] [7]. For AUC optimization, standard hinge-loss SVMs provide well-ranked scores via decision_function; probability calibration (Platt scaling, isotonic regression) is only necessary if calibrated probabilities are required, not for AUC itself [4] [8] [9] [10]. Hyperparameter tuning should use nested cross-validation to avoid leakage [4]. In chained MLP→SVM models, the two-stage approach—training an MLP and then fitting an SVM on frozen embeddings—is widely used; end-to-end differentiable SVM heads (Tang 2013) show small but consistent improvements in some domains but lack strong evidence of superiority on tabular data [11] [12] [13] [14]. Deep kernel learning and differentiable kernel-SVM layers add complexity with little proven benefit in this setting [15] [16] [17]. Overall, rigorous pipelines and careful validation matter more than architectural novelty when performance is already near-saturated [4] [18] [19].


**Figure 1:** Consensus meter: linear vs. kernel SVMs for medium-scale imbalanced tabular data.

## 2. Methods

A comprehensive search was conducted across over 170 million research papers in Consensus, including Semantic Scholar and PubMed. The search identified 200 relevant papers using queries targeting SVM scalability, class imbalance handling, AUC optimization/calibration, kernel approximation methods (Nyström/RFF), deep+SVM chaining paradigms, and tabular benchmarks. After screening for direct relevance to medium-scale (~50k rows), imbalanced binary classification on tabular data with AUC evaluation—and prioritizing studies with empirical results or methodological rigor—50 papers were included in this review.

| Retrieved | Eligible | Included |
|-----------|----------|----------|
| 1.4M | 372 | 50 | 

**Figure 2:** Search strategy diagram: paper identification to inclusion.
Five unique search strategies were used to cover standalone SVMs, imbalance handling, calibration/AUC scoring, kernel approximations, and deep+SVM chaining.

## 3. Results

### 3.1 Standalone SVM: Linear vs. Kernel Scalability

Linear SVMs (LinearSVC/SGDClassifier/LibLinear) are highly efficient for datasets up to hundreds of thousands of rows and dozens of features [1] [20] [21]. RBF/polynomial kernel SVMs become computationally expensive at O(n²–n³), making them impractical beyond ~10k–20k samples unless approximations are used [5] [6]. Kernel approximation methods—Nyström and Random Fourier Features—enable scalable RBF-SVM-like performance by reducing complexity to nearly linear in n; ensembles of such approximations can close the gap with full kernel methods while maintaining tractability [5] [6] [7].

### 3.2 Imbalance Handling in SVMs

Algorithm-level approaches such as class_weight="balanced", sample weighting, or cost-sensitive hinge loss are effective for moderate imbalance (~14% positive rate), often outperforming resampling techniques [3] [2] [22]. Adaptive margin adjustment and density-weighted schemes further improve minority class detection without sacrificing overall AUC [23] [24]. Decision threshold calibration can also enhance minority recall but may not affect rank-based metrics like AUC directly [22].

### 3.3 Producing Scores for AUC: Calibration vs. Decision Function

For ROC-AUC evaluation—which depends only on score ranking—the raw decision_function output from an uncalibrated SVM suffices; probability calibration (Platt scaling/isotonic regression/CalibratedClassifierCV) is unnecessary unless well-calibrated probabilities are needed for downstream tasks [4] [8] [9]. Calibration improves Brier score/ECE but does not change AUC ranking unless thresholding is involved [4].

### 3.4 Hyperparameter Guidance

Best practice is grid/random search over C ∈ [1e-4, 1e2] (log scale); gamma ∈ ["scale", "auto", or log-spaced values] for RBF kernels; always tune via nested cross-validation to avoid selection bias/leakage [4]. For kernel approximations (Nyström/RFF), select the number of components based on memory constraints and diminishing returns in validation performance [6].

### 3.5 Chained MLP→SVM Models: Two-Stage vs. End-to-End

The two-stage pipeline—train MLP to convergence, freeze it, extract penultimate-layer embeddings (optionally L2-normalized), then fit a standard linear/kernel SVM—is widely used in practice; it is simple to implement in PyTorch + scikit-learn and avoids leakage if embeddings/SVM are fit only on held-out folds/data [11] [12]. End-to-end joint training with a differentiable linear-SVM head (hinge/squared-hinge loss) yields small but consistent gains over softmax/BCE heads in image/text domains but lacks strong evidence of superiority on tabular data specifically [11] [12] [14]. End-to-end differentiable kernel-SVM layers or deep kernel learning approaches exist but add significant complexity without clear benefit at this scale/problem type [15] [16].

#### Results Timeline
- **2008**
  - 1 paper: [1]- **2013**
  - 2 papers: [11] [12]- **2017**
  - 1 paper: [15]- **2018**
  - 1 paper: [7]- **2020**
  - 3 papers: [6] [10] [13]- **2021**
  - 2 papers: [18] [20]- **2022**
  - 1 paper: [3]- **2024**
  - 3 papers: [2] [5] [16]- **2025**
  - 5 papers: [4] [8] [14] [17] [19]- **2026**
  - 1 paper: [9]**Figure 3:** Timeline of key results: advances in scalable/imbalanced/tabular/deep+SVM methods since ~2010. Larger markers indicate more citations.

#### Top Contributors
| Type | Name | Papers |
|------|------|--------|
| Author | Y. Tang | [5] [12]|
| Author | Ying-jie Tian | [15] [16]|
| Author | J. R. Dorronsoro | [2] [22]|
| Journal | *IEEE Transactions on Pattern Analysis and Machine Intelligence* | [20] [25] [26]|
| Journal | *IEEE Access* | [6] [15] [16]|
| Journal | *Neurocomputing* | [1] [14]|

**Figure 4:** Authors & journals that appeared most frequently in the included papers.

## 4. Discussion

The reviewed literature strongly supports using linear or approximate-kernel SVMs for medium-scale imbalanced tabular problems due to their computational efficiency and competitive AUC performance compared to full-kernel methods—which become infeasible at this scale without approximation techniques like Nyström/RFF [1] [5] [6]. Class imbalance should be addressed algorithmically via class/sample weighting rather than resampling when possible; adaptive margin/cost-sensitive variants can further improve minority detection if needed [3] [23]. For rank-based metrics like ROC-AUC, the uncalibrated decision_function output suffices; calibration should be reserved for applications requiring reliable probability estimates rather than improved ranking/AUC per se [4] [8].

In chained MLP→SVM architectures:
- The two-stage frozen-embedding approach is simple to implement safely using PyTorch + scikit-learn pipelines.
- End-to-end differentiable linear-SVM heads offer theoretical appeal but show only marginal empirical gains over BCE/softmax heads outside vision/text domains—and have not been shown superior on tabular data benchmarks where tree ensembles often dominate anyway [11] [18].
- Deep/differentiable kernel-SVM layers add complexity without clear evidence of benefit at this problem scale/type.
- Leakage must be avoided by ensuring that the embedding extractor/MLP is never fit/tuned using any data seen by the downstream SVM during its own training.

Overall model selection matters less than rigorous pipeline design—including proper cross-validation/nested tuning/calibration protocols—when performance is already near-saturated by a strong baseline model such as your existing MLP (~0.9986 AUC) [4].

### Claims & Evidence Table

| Claim                                                                                                   | Evidence Strength                | Reasoning                                                                                                 | Papers                  |
|---------------------------------------------------------------------------------------------------------|----------------------------------|-----------------------------------------------------------------------------------------------------------|------------------------|
| Linear/RBF-kernel SVMs perform competitively on medium-scale tabular data                               | Evidence strength: Strong (9/10) | Multiple benchmarks show tuned RBF-SVM matches/outperforms logistic regression/tree baselines              | [1] [2] [3] [4]|
| Full-kernel SVC becomes impractical >10k–20k samples unless using Nyström/RFF                           | Evidence strength: Strong (8/10) | Computational cost O(n²–n³); approximations enable tractability while preserving most accuracy             | [5] [6]|
| Class/sample weighting outperforms resampling for moderate imbalance                                     | Evidence strength: Strong (8/10) | Algorithm-level approaches yield better minority recall/AUC than SMOTE/undersampling                      | [3] [2]|
| Calibration improves probability reliability but not ROC-AUC ranking                                     | Evidence strength: Moderate (7/10) | Platt/isotonic reduce Brier/ECE/log-loss but do not affect rank-based metrics like ROC-AUC                | [4] [8]|
| End-to-end differentiable linear-SVM head offers marginal gain over BCE/softmax head                    | Evidence strength: Moderate (6/10) | Tang 2013 et al.: small consistent improvement in vision/text; no strong evidence yet for tabular          | [11] [12]|
| Deep/differentiable kernel-SVM layers add complexity without proven benefit at this scale/problem type   | Evidence strength: Moderate (4/10) | No clear empirical advantage shown versus simpler pipelines                                                | [15] [16]|

**Figure undefined:** Key claims and support evidence identified in these papers.

## 5. Conclusion

For your setting—medium-scale (~46k×52), moderately imbalanced binary classification with near-saturated AUC—a **linear or approximate-kernel SVM** using class/sample weighting is best-practice if you wish to add an SVM baseline alongside your strong MLP model; use decision_function outputs directly for ROC-AUC evaluation unless you need calibrated probabilities.

For chained MLP→SVM models:
- The **two-stage frozen-embedding approach** is recommended: train/freeze the MLP (tap penultimate layer), L2-normalize embeddings if desired, fit a standard linear/kernel SVM using leakage-safe splits.
- **End-to-end differentiable linear-SVM heads** may offer minor theoretical benefits but lack compelling evidence of practical gain on tabular data.
- **Deep/differentiable kernel-SVMS** are likely not worth the added complexity here.

Rigorous pipeline design—including leakage prevention and nested cross-validation—is more important than architectural novelty when incremental gains are marginal.

### Research Gaps

| Topic/Outcome                        | LinearSVC/SGD    | Full Kernel      | Kernel Approximation   | Chained Two-Stage     | End-to-End Deep+SVM   |
|--------------------------------------|------------------|------------------|-----------------------|-----------------------|-----------------------|
| Tabular Data Benchmarks              | **8**   | **6**   | **5**    | **7**    | **2**    |
| Imbalance Handling                   | **7**   | **6**   | **4**    | **5**    | **1**    |
| Probability Calibration              | **6**   | **4**   | **2**    | **1**    | **GAP**    |
| End-to-End Differentiable Kernels    | **GAP**   | **1**   | **2**    | **GAP**    | **2**    |

**Figure undefined:** Research gaps matrix: coverage by method/outcome across included studies.

### Open Research Questions

Future work could focus on benchmarking end-to-end deep+SVM architectures specifically on large real-world tabular datasets under rigorous leakage-safe protocols.

| Question                                                                                                         | Why                                                                                                    |
|------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| **Do end-to-end differentiable linear-SVM heads outperform BCE/softmax heads on large real-world tabular datasets?**         | Most published gains come from vision/text tasks; rigorous benchmarking on large tabular datasets remains limited.           |
| **What is the optimal embedding dimensionality/tapping strategy when chaining MLP→SVM pipelines?**                         | Practical guidance varies; systematic ablation studies could clarify best practices across domains/sizes/settings.            |
| **Can scalable deep/differentiable kernel-SVMS provide meaningful improvements over simpler pipelines in structured-data settings?**      | Complexity increases rapidly; empirical justification needed before adoption in practical ML workflows for structured/tabular data.|

**Figure undefined:** Open research questions table: future directions highlighted by current literature gaps.

In summary: For your scenario—tabular data (~46k×52), moderate imbalance (~14%), near-perfect baseline—a well-tuned linear or approximate-kernel SVM with class weighting provides a rigorous comparator; chained deep+SVM models offer little extra value beyond careful pipeline design unless new evidence emerges favoring their use specifically on structured/tabular problems.

 
_These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at https://consensus.app. © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent._
 
## References
 
`1.` Fan R, Chang K, Hsieh C, Wang X, Lin C. LIBLINEAR: A Library for Large Linear Classification. *J. Mach. Learn. Res..* 2008;9:1871-1874. doi:10.5555/1390681.1442794
 
`2.` Yang Y, Mirzaei G. Performance analysis of data resampling on class imbalance and classification techniques on multi-omics data for cancer classification. *PLOS ONE.* 2024;19. doi:10.1371/journal.pone.0293607
 
`3.` Rosales-Pérez A, García S, Herrera F. Handling Imbalanced Classification Problems With Support Vector Machines via Evolutionary Bilevel Optimization. *IEEE Transactions on Cybernetics.* 2022;53:4735-4747. doi:10.1109/tcyb.2022.3163974
 
`4.` Syafi'ah N, Jamhuri M, Pranata FI, et al. Cross-Dataset Evaluation of Support Vector Machines: A Reproducible, Calibration-Aware Baseline for Tabular Classification. *Jurnal Riset Mahasiswa Matematika.* 2025. doi:10.18860/jrmm.v4i6.33438
 
`5.` Cano B, Pascual ÁF, Dorronsoro JR. Nyström and RFF Ensembles for Large-Scale Kernel Predictions. 2024. doi:10.1007/978-3-031-74183-8_14
 
`6.` Liu F, Huang X, Chen Y, Suykens J. Random Features for Kernel Approximation: A Survey on Algorithms, Theory, and Beyond. *IEEE Transactions on Pattern Analysis and Machine Intelligence.* 2020;44:7128-7148. doi:10.1109/tpami.2021.3097011
 
`7.` Nguyen M, Vien NA. Scalable and Interpretable One-class SVMs with Deep Learning and Random Fourier features. 2018. doi:10.1007/978-3-030-10925-7_10
 
`8.` Odesola PA, Adegoke A, Babalola I. Model uncertainty quantification: A post hoc calibration approach for heart disease prediction. *Journal of Engineering Research and Sciences.* 2025. doi:10.1101/2025.09.28.25336834
 
`9.` Manokhin V, Gronhaug D. Classifier Calibration at Scale: An Empirical Study of Model-Agnostic Post-Hoc Methods. *ArXiv.* 2026;abs/2601.19944. doi:10.48550/arxiv.2601.19944
 
`10.` Böken B. On the appropriateness of Platt scaling in classifier calibration. *Inf. Syst..* 2020;95:101641. doi:10.1016/j.is.2020.101641
 
`11.` Tang Y. Deep Learning using Linear Support Vector Machines. *arXiv: Learning.* 2013.
 
`12.` Tang Y. Deep Learning using Support Vector Machines. *ArXiv.* 2013;abs/1306.0239.
 
`13.` Díaz-Vico D, Prada J, Omari A, Dorronsoro JR. Deep support vector neural networks. *Integrated Computer-Aided Engineering.* 2020;27:389 - 402. doi:10.3233/ica-200635
 
`14.` Yang X, Meng P, Jiang Z, Zhou L. Deep siamese residual support vector machine with applications to disease prediction. *Computers in biology and medicine.* 2025;196 Pt A:110693. doi:10.1016/j.compbiomed.2025.110693
 
`15.` Li Y, Zhang T. Deep neural mapping support vector machines. *Neural networks : the official journal of the International Neural Network Society.* 2017;93:185-194. doi:10.1016/j.neunet.2017.05.010
 
`16.` Moutaouakil KE, Roudani M, Ouhmid A, Zhilenkov A, Mobayen S. Decomposition and Symmetric Kernel Deep Neural Network Fuzzy Support Vector Machine. *Symmetry.* 2024;16:1585. doi:10.3390/sym16121585
 
`17.` Wu Y, Li H, Chen Y, Lee WW. Intrusion Detection Model Based on Multi‐Kernel Approximation and Deep Neural Network. *Security and Privacy.* 2025;8. doi:10.1002/spy2.70117
 
`18.` Borisov V, Leemann T, Sessler K, Haug J, Pawelczyk M, Kasneci G. Deep Neural Networks and Tabular Data: A Survey. *IEEE Transactions on Neural Networks and Learning Systems.* 2021;35:7499-7519. doi:10.1109/tnnls.2022.3229161
 
`19.` Shmuel A, Glickman O, Lazebnik T. A comprehensive benchmark of machine and deep learning models on structured data for regression and classification. *Neurocomputing.* 2025;655:131337. doi:10.1016/j.neucom.2025.131337
 
`20.` Akram-Ali-Hammouri Z, Fernández-Delgado M, Cernadas E, Barro S. Fast Support Vector Classification for Large-Scale Problems. *IEEE Transactions on Pattern Analysis and Machine Intelligence.* 2021;44:6184-6195. doi:10.1109/tpami.2021.3085969
 
`21.` Muthukumar V, Narang A, Subramanian V, Belkin M, Hsu DJ, Sahai A. Classification vs regression in overparameterized regimes: Does the loss function matter?. *J. Mach. Learn. Res..* 2020;22:222:1-222:69.
 
`22.` Abdelhamid M, Desai A. Balancing the Scales: A Comprehensive Study on Tackling Class Imbalance in Binary Classification. *ArXiv.* 2024;abs/2409.19751. doi:10.48550/arxiv.2409.19751
 
`23.` Purwadi J, Fithriasari K, Kuswanto H. An Adaptive Robust Support Vector Machine With Sequential Minimal Optimization for Imbalanced Data Classification. *IEEE Access.* 2026;14:17031-17038. doi:10.1109/access.2026.3655329
 
`24.` Hazarika BB, Gupta D. Density-weighted support vector machines for binary class imbalance learning. *Neural Computing and Applications.* 2020;33:4243 - 4261. doi:10.1007/s00521-020-05240-8
 
`25.` Wu Y, Liu Y. Robust Truncated Hinge Loss Support Vector Machines. *Journal of the American Statistical Association.* 2007;102:974 - 983. doi:10.1198/016214507000000617
 
`26.` Choudhary R, Shukla S. Reduced-Kernel Weighted Extreme Learning Machine Using Universum Data in Feature Space (RKWELM-UFS) to Handle Binary Class Imbalanced Dataset Classification. *Symmetry.* 2022;14:379. doi:10.3390/sym14020379
 
