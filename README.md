## A Three-Phase Deep Learning Framework for Automated Infarct Segmentation in Preclinical Stroke Imaging

Public Snippets Only. This repository intentionally publishes selected fragments (model skeletons and minimal training interfaces) for U-Net, Deep U-Net, RA-UNet, and ResNet-50–based fine-tuning  to support peer review and reproducibility claims.  

### Models Covered (snippet-only)
- U-Net – deeper encoder/decoder scaffold (dropouts, BN, skip wiring withheld)
- RA-UNet– residual attention gate interfaces (attention/gating internals withheld)
- U-Net (ResNet-50 encoder) – fine-tuning interface (decoder & training loop withheld)
- RA-UNet (ResNet-50 encoder) – fine-tuning interface (attention & training loop withheld)


## Baseline Training (Snippet-Only)-Phae-I
- U-Net snippets: `models/unet/snippets.py`  
  Runner: `src/train_unet_snippet.py`  
  Config: `configs/unet_snippet.yaml`

- RA-UNet snippets: `models/raunet/snippets.py`  
  Runner: `src/train_raunet_snippet.py`  
  Config: `configs/raunet_snippet.yaml`

## Fine-tuning (Snippet-Only)
- U-Net (ResNet-50 encoder): `models/unet/finetune_resnet50_snippet.py`  
  Runner: `src/finetune_unet_resnet50_snippet.py`  
  Config: `configs/unet_resnet50_finetune.yaml`

- RA-UNet (ResNet-50 encoder): `models/raunet/finetune_resnet50_snippet.py`  
  Runner: `src/finetune_raunet_resnet50_snippet.py`  
  Config: `configs/raunet_resnet50_finetune.yaml`

# Full Factorial design of experiment based model development (Phase -II)
Deep U-Net (5-level / 9-layer, Full-Factorial DOE)
- Model file: `models/unet/deep_unet_snippet.py`  
- Runner: `src/train_deep_unet_snippet.py`  
- Config:`configs/deep_unet_snippet.yaml`

Note: Full implementations (data pipeline, complete training loops, augmentation policy, metrics, callbacks incl. ECE, and evaluation) are withheld and available upon request under the Custom Research License.

# References
-An, J., Wendt, L., Wiese, G., Herold, T., Rzepka, N., Mueller, S., Koch, S.P., Hoffmann, C.J., Harms, C., Boehm-Sturm, P., 2023. Deep learning-based automated lesion segmentation on mouse stroke magnetic resonance images. Sci Rep 13. https://doi.org/10.1038/s41598-023-39826-8
-Arora, A., Alderman, J.E., Palmer, J., Ganapathi, S., Laws, E., McCradden, M.D., Oakden-Rayner, L., Pfohl, S.R., Ghassemi, M., McKay, F., Treanor, D., Rostamzadeh, N., Mateen, B., Gath, J., Adebajo, A.O., Kuku, S., Matin, R., Heller, K., Sapey, E., Sebire, N.J., Cole-Lewis, H., Calvert, M., Denniston, A., Liu, X., 2023. The value of standards for health datasets in artificial intelligence-based applications. Nat Med 29, 2929–2938. https://doi.org/10.1038/s41591-023-02608-w
-Athanasiou, G., Arcos, J.L., Cerquides, J., 2023. Enhancing Medical Image Segmentation: Ground Truth Optimization through Evaluating Uncertainty in Expert Annotations. Mathematics 11. https://doi.org/10.3390/math11173771
-Babar, M., Qureshi, B., Koubaa, A., 2024. Investigating the impact of data heterogeneity on the performance of federated learning algorithm using medical imaging. PLoS One 19. https://doi.org/10.1371/journal.pone.0302539
-Bilal, M., Podishetti, R., Koval, L., Gaafar, M.A., Grossmann, D., Bregulla, M., 2024. The Effect of Annotation Quality on Wear Semantic Segmentation by CNN. Sensors 24. https://doi.org/10.3390/s24154777
-Cabeen, R.P., Mandeville, J., Hyder, F., Sanganahalli, B.G., Thedens, D.R., Arbab, A.S., Huang, S., Bibic, A., Tarakci, E., Mihailovic, J., Morais, A., Lamb, J., Nagarkatti, K., Toga, A.W., Lyden, P., Ayata, C., 2023. Computational Image-Based Stroke Assessment for Evaluation of Cerebroprotectants with Longitudinal and Multi-Site Preclinical MRI, in: 2023 IEEE 20th International Symposium on Biomedical Imaging (ISBI). pp. 1–5. https://doi.org/10.1109/ISBI53787.2023.10230408
-Cerqueira, V., Santos, M., Roque, L., Baghoussi, Y., Soares, C., 2025. Online Data Augmentation for Forecasting with Deep Learning.
-Charisis, C., Argyropoulos, D., 2024. Deep learning-based instance segmentation architectures in agriculture: A review of the scopes and challenges. Smart Agricultural Technology. https://doi.org/10.1016/j.atech.2024.100448
-Chen, J., Mei, J., Li, X., Lu, Y., Yu, Q., Wei, Q., Luo, X., Xie, Y., Adeli, E., Wang, Y., Lungren, M.P., Zhang, S., Xing, L., Lu, L., Yuille, A., Zhou, Y., 2024. TransUNet: Rethinking the U-Net architecture design for medical image segmentation through the lens of transformers. Med Image Anal 97. https://doi.org/10.1016/j.media.2024.103280
-Choudhury, A., Asan, O., 2020. Role of artificial intelligence in patient safety outcomes: Systematic literature review. JMIR Med Inform. https://doi.org/10.2196/18599
-Davila Delgado, J.M., Oyedele, L., 2021. Deep learning with small datasets: using autoencoders to address limited datasets in construction management. Appl Soft Comput 112. https://doi.org/10.1016/j.asoc.2021.107836
-Ehab, W., Huang, L., Li, Y., 2024. UNet and Variants for Medical Image Segmentation. International Journal of Network Dynamics and Intelligence 100009. https://doi.org/10.53941/ijndi.2024.100009
-El-Taraboulsi, J., Cabrera, C.P., Roney, C., Aung, N., 2023. Deep neural network architectures for cardiac image segmentation. Artificial Intelligence in the Life Sciences 4. https://doi.org/10.1016/j.ailsci.2023.100083
-Faria, A. V., Joel, S.E., Zhang, Y., Oishi, K., van Zjil, P.C.M., Miller, M.I., Pekar, J.J., Mori, S., 2012. Atlas-based analysis of resting-state functional connectivity: -Evaluation for reproducibility and multi-modal anatomy-function correlation studies. Neuroimage 61, 613–621. https://doi.org/10.1016/j.neuroimage.2012.03.078
-Faryna, K., van der Laak, J., Litjens, G., 2024. Automatic data augmentation to improve generalization of deep learning in H&E stained histopathology. Comput Biol Med 170. https://doi.org/10.1016/j.compbiomed.2024.108018
-Hao, R., Namdar, K., Liu, L., Haider, M.A., Khalvati, F., 2021. A Comprehensive Study of Data Augmentation Strategies for Prostate Cancer Detection in Diffusion-Weighted MRI Using Convolutional Neural Networks. J Digit Imaging 34, 862–876. https://doi.org/10.1007/s10278-021-00478-7
-Huang, Y., Leotta, N.J., Hirsch, L., Gullo, R. Lo, Hughes, M., Reiner, J., Saphier, N.B., Myers, K.S., Panigrahi, B., Ambinder, E., Di Carlo, P., Grimm, L.J., Lowell, D., Yoon, S., Ghate, S. V., Parra, L.C., Sutton, E.J., 2025. Cross-site Validation of AI Segmentation and Harmonization in Breast MRI. Journal of Imaging Informatics in Medicine 38, 1642–1652. https://doi.org/10.1007/s10278-024-01266-9
-Kamel, P., Kanhere, A., Kulkarni, P., Khalid, M., Steger, R., Bodanapally, U., Gandhi, D., Parekh, V., Yi, P.H., 2024. Optimizing Acute Stroke Segmentation on MRI Using Deep Learning: Self-Configuring Neural Networks Provide High Performance Using Only DWI Sequences. Journal of Imaging Informatics in Medicine. https://doi.org/10.1007/s10278-024-00994-2
-Karimi, D., Warfield, S.K., Gholipour, A., 2021. Transfer learning in medical image segmentation: New insights from analysis of the dynamics of model parameters and learned representations. Artif Intell Med 116. https://doi.org/10.1016/j.artmed.2021.102078
-Khor, H.G., Ning, G., Sun, Y., Lu, X., Zhang, X., Liao, H., 2023. Anatomically constrained and attention-guided deep feature fusion for joint segmentation and deformable medical image registration. Med Image Anal 88. https://doi.org/10.1016/j.media.2023.102811
-Koçak, B., Ponsiglione, A., Stanzione, A., Bluethgen, C., Santinha, J., Ugga, L., Huisman, M., Klontzas, M.E., Cannella, R., Cuocolo, R., 2024. Bias in artificial intelligence for medical imaging: fundamentals, detection, avoidance, mitigation, challenges, ethics, and prospects. Diagnostic and Interventional Radiology. https://doi.org/10.4274/dir.2024.242854
-Kugelman, J., Allman, J., Read, S.A., Vincent, S.J., Tong, J., Kalloniatis, M., Chen, F.K., Collins, M.J., Alonso-Caneiro, D., 2022. A comparison of deep learning U-Net architectures for posterior segment OCT retinal layer segmentation. Sci Rep 12. https://doi.org/10.1038/s41598-022-18646-2
-Küper, A., Blanc-Durand, P., Gafita, A., Kersting, D., Fendler, W.P., Seibold, C., Moraitis, A., Lückerath, K., James, M.L., Seifert, R., 2023. Is There a Role of Artificial Intelligence in Preclinical Imaging? Semin Nucl Med. https://doi.org/10.1053/j.semnuclmed.2023.03.003
-Lan, J., Chen, M., Wang, J., Du, M., Wu, Z., Zhang, H., Xue, Y., Wang, T., Chen, L., Xu, C., Han, Z., Hu, Z., Zhou, Y., Zhou, X., Tong, T., Chen, G., 2023a. Using less annotation workload to establish a pathological auxiliary diagnosis system for gastric cancer. Cell Rep Med 4. https://doi.org/10.1016/j.xcrm.2023.101004
-Lan, J., Chen, M., Wang, J., Du, M., Wu, Z., Zhang, H., Xue, Y., Wang, T., Chen, L., Xu, C., Han, Z., Hu, Z., Zhou, Y., Zhou, X., Tong, T., Chen, G., 2023b. Using less annotation workload to establish a pathological auxiliary diagnosis system for gastric cancer. Cell Rep Med 4. https://doi.org/10.1016/j.xcrm.2023.101004
-Liu, X., Ono, K., Bise, R., 2024. A data augmentation approach that ensures the reliability of foregrounds in medical image segmentation. Image Vis Comput 147. https://doi.org/10.1016/j.imavis.2024.105056
-Lourbopoulos, A., Mourouzis, I., Xinaris, C., Zerva, N., Filippakis, K., Pavlopoulos, A., Pantos, C., 2021. Translational Block in Stroke: A Constructive and “Out-of-the-Box” Reappraisal. Front Neurosci 15. https://doi.org/10.3389/fnins.2021.652403
-Luo, J., Dai, P., He, Z., Huang, Z., Liao, S., Liu, K., 2024. Deep learning models for ischemic stroke lesion segmentation in medical images: A survey. Comput Biol Med. https://doi.org/10.1016/j.compbiomed.2024.108509
-Lyden, P.D., Diniz, M.A., Bosetti, F., Lamb, J., Nagarkatti, K.A., Rogatko, A., Kim, S., Cabeen, R.P., Koenig, J.I., Akhter, K., Arbab, A.S., Avery, B.D., Beatty, H.E., Bibic, A., Cao, S., Simoes, L., Boisserand, B., Chamorro, A., Chauhan, A., Diaz-Perez, S., Dhandapani, K., Dhanesha, N., Goh, A., Herman, A.L., Hyder, F., Imai, T., Johnson, -C.W., Khan, M.B., Kamat, P., Karuppagounder, S.S., Kumskova, M., Mihailovic, J.M., Mandeville, J.B., Morais, A., Patel, R.B., Sanganahalli, B.G., Smith, C., Shi, Y., Sutariya, B., Thedens, D., Qin, T., Velazquez, S.E., Aronowski, J., Ayata, C., Chauhan, A.K., Leira, E.C., Hess, D.C., Koehler, R.C., Mccullough, L.D., Sansing, L.H., 2023. A multi-laboratory preclinical trial in rodents to assess treatment candidates for acute ischemic stroke.
-Mahajan, K.R., Ontaneda, D., 2017. The Role of Advanced Magnetic Resonance Imaging Techniques in Multiple Sclerosis Clinical Trials. Neurotherapeutics. https://doi.org/10.1007/s13311-017-0561-8
-Mzoughi, H., Njeh, I., Slima, M. Ben, Ben Hamida, A., Mhiri, C., Mahfoudh, K. Ben, 2021. Towards a computer aided diagnosis (CAD) for brain MRI glioblastomas tumor exploration based on a deep convolutional neuronal networks (D-CNN) architectures. Multimed Tools Appl 80, 899–919. https://doi.org/10.1007/s11042-020-09786-6
-Nagarkatti, K., Diniz, M., Cabeen, R., Estrada, M., Crawford, K., Rogatko, A., Kim, S., Ayata, C., Chauhan, A., Hess, D., Khan, M., Kumskova, M., Leira, E., Patel, R., Lyden, P., Lamb, J., 2025. Methods for randomized, blinded, controlled evaluation of putative disease interventions in multi-laboratory, preclinical assessment networks. https://doi.org/10.21203/rs.3.rs-3054771/v1
-Ni, R., Han, K., Haibe-Kains, B., Rink, A., 2024. Generalizability of deep learning in organ-at-risk segmentation: A transfer learning study in cervical brachytherapy. Radiotherapy and Oncology 197. https://doi.org/10.1016/j.radonc.2024.110332
-Puzio, T., Matera, K., Karwowski, J., Piwnik, J., Białkowski, S., Podyma, M., Dunikowski, K., Siger, M., Stasiołek, M., Grzelak, P., Bobeff, E.J., 2025. Deep learning-based automatic segmentation of brain structures on MRI: A test-retest reproducibility analysis. Comput Struct Biotechnol J 28, 128–140. https://doi.org/10.1016/j.csbj.2025.04.007
-Razavi, M., Mavaddati, S., Koohi, H., 2024. ResNet deep models and transfer learning technique for classification and quality detection of rice cultivars. Expert Syst Appl 247. https://doi.org/10.1016/j.eswa.2024.123276
-Renard, F., Guedria, S., Palma, N. De, Vuillerme, N., 2020a. Variability and reproducibility in deep learning for medical image segmentation. Sci Rep 10. https://doi.org/10.1038/s41598-020-69920-0
-Renard, F., Guedria, S., Palma, N. De, Vuillerme, N., 2020b. Variability and reproducibility in deep learning for medical image segmentation. Sci Rep 10. https://doi.org/10.1038/s41598-020-69920-0
-Schutera, M., Rettenberger, L., Pylatiuk, C., Reischl, M., 2022. Methods for the frugal labeler: Multi-class semantic segmentation on heterogeneous labels. PLoS One 17. https://doi.org/10.1371/journal.pone.0263656
-Sylolypavan, A., Sleeman, D., Wu, H., Sim, M., 2023. The impact of inconsistent human annotations on AI driven clinical decision making. NPJ Digit Med 6. https://doi.org/10.1038/s41746-023-00773-3
-Werdiger, F., Yogendrakumar, V., Visser, M., Kolacz, J., Lam, C., Hill, M., Chen, C., Parsons, M.W., Bivard, A., 2024. Clinical performance review for 3-D Deep Learning segmentation of stroke infarct from diffusion-weighted images. Neuroimage: Reports 4. https://doi.org/10.1016/j.ynirp.2024.100196
-Wołczyk, M., Cupiał, B., Ostaszewski, M., Bortkiewicz, M., Zając, M., Pascanu, R., Kuciński, Ł., Miłoś, P., 2024. Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem.
-Xiao, R., Li, Z., Miao, X., Wang, W., Zhang, P., 2022. GuidedMix: An on-the-fly data augmentation approach for robust speaker recognition system. Electron Lett. https://doi.org/10.1049/ell2.12354
-Xiao, S., Wang, P., Diao, W., Rong, X., Li, X., Fu, K., Sun, X., 2023. MoCG: Modality Characteristics-Guided Semantic Segmentation in Multimodal Remote Sensing Images. IEEE Transactions on Geoscience and Remote Sensing 61. https://doi.org/10.1109/TGRS.2023.3334471
-Yu, X., Zhao, H., Zhang, M., Wei, Y., Zhou, L., Ou, L., 2024. DynamicAug: Enhancing Transfer Learning Through Dynamic Data Augmentation Strategies Based on Model State. Neural Process Lett 56. https://doi.org/10.1007/s11063-024-11626-9
-Zhang, C., Bao, N., Sun, H., Li, H., Li, J., Qian, W., Zhou, S., 2022. A Deep Learning Image Data Augmentation Method for Single Tumor Segmentation. Front Oncol 12. https://doi.org/10.3389/fonc.2022.782988
-Zhou, B., Augenfeld, Z., Chapiro, J., Zhou, S.K., Liu, C., Duncan, J.S., 2021. Anatomy-guided multimodal registration by learning segmentation without ground truth: Application to intraprocedural CBCT/MR liver segmentation and registration. Med Image Anal 71. https://doi.org/10.1016/j.media.2021.102041
-Zhou, S.K., Greenspan, H., Davatzikos, C., Duncan, J.S., Van Ginneken, B., Madabhushi, A., Prince, J.L., Rueckert, D., Summers, R.M., 2021. A Review of Deep Learning in Medical Imaging: Imaging Traits, Technology Trends, Case Studies with Progress Highlights, and Future Promises. Proceedings of the IEEE 109, 820–838. https://doi.org/10.1109/JPROC.2021.3054390

