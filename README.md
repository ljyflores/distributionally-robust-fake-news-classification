# DRO
<b>TLDR:</b> We adapt BERT for Classification on the LIAR fake-news classification dataset, but optimize it using a distributionally robust optimization (DRO) algorithm by Sagawa et al. (2020). 

The writeup explaining DRO and showing our results is linked <a href="https://drive.google.com/file/d/1oZH67nraRTS5kXUM_nrTc0JiyF4bWEwI/view?usp=sharing">here</a>, and the models are saved <a href="https://drive.google.com/drive/folders/1wiEwUTLBuPDI8fj_GfYPWnl9gn81tovh?usp=sharing">here</a>.

Our work adapts the algorithm proposed by Sagawa et al. (2020), whose paper is linked <a href="https://arxiv.org/pdf/1911.08731.pdf">here</a>.

If this was useful, please consider citing the original work:
```
@misc{https://doi.org/10.48550/arxiv.1911.08731,
  doi = {10.48550/ARXIV.1911.08731},
  url = {https://arxiv.org/abs/1911.08731},
  author = {Sagawa, Shiori and Koh, Pang Wei and Hashimoto, Tatsunori B. and Liang, Percy},
  keywords = {Machine Learning (cs.LG), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization},
  publisher = {arXiv},
  year = {2019},
  copyright = {arXiv.org perpetual, non-exclusive license}
}

```
