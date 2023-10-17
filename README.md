<div align="center">
   <centre><h1>EPISTASIS DETECTION</centre><br />
      </div>

Epistasis Detection is a bioinformatics application which refers to the discovery of correlations between Single Nucleotide Polymorphisms (SNPs) and an observed phenotype (e.g., a disease). Herein, a machine learning application is developed for high order epistasis detection.

<h2>ABOUT</h2>

**Importance of Epistasis Detection**

Genome-Wide Association Studies (GWAS) analyze the influence of individual genetic markers on well-known diseases. However, this approach ignores gene-gene interactions (epistasis), which are of utmost importance to understand complex diseases. Therefore, finding new SNP associations has a high impact on our understanding of these diseases, as well as precision medicine, contributing to improve personalized healthcare.

**Why Machine Learning?**

The optimal strategy for epistasis detection is to evaluate all possible SNP combination, but this incurs in a high computational cost (e.g., on a dataset with 500 000 SNPs, evaluating 2-SNP associations would amount to analyze 125 billion combinations). A machine learning model is agnostic to interaction orders and does not need to search for SNP combinations. A model can be trained on an epistasis dataset and its results are afterwards interpreted to find possible SNP interactions to the studied disease.

<h2>TECHNOLOGIES USED</h2>
<p align="center">
  <img src="https://github.com/Miguel491/episdet-transformer/blob/main/InteloneAPI.jpg" width="400" height="240" >
</p>

The [Intel® OneAPI toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#gs.3btkxe), [Intel® OpenVINO toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html) enables models to be trained in a more efficient way and results in faster training times.

<h2>METHODOLOGY</h2>
A transformer neural network is implemented for high order epistasis detection. Interpretation metrics (e.g., attention scores) are analyzed post-training to identify SNPs that may be relevant to the observed phenotype. The model is trained on simulated datasets to test its capabilities for epistasis detection.
