---
license: cc-by-nc-sa-4.0
task_categories:
- tabular-classification
pretty_name: fairjob
size_categories:
- 1M<n<10M
---
# CRITEO FAIRNESS IN JOB ADS DATASET

## Summary

This dataset is released by Criteo to foster research and innovation on Fairness in Advertising and AI systems in general. 
See also [Criteo pledge for Fairness in Advertising](https://fr.linkedin.com/posts/diarmuid-gill_advertisingfairness-activity-6945003669964660736-_7Mu).

The dataset is intended to learn click predictions models and evaluate by how much their predictions are biased between different gender groups. 

## License

The data is released under the [CC-BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/) 4.0 license. 
You are free to Share and Adapt this data provided that you respect the Attribution, NonCommercial and ShareAlike conditions. 
Please read carefully the full license before using. 

## Data description
The dataset contains pseudononymized users' context and publisher features that was collected from a job targeting campaign ran for 5 months by Criteo AdTech company.  Each line represents a product that was shown to a user. Each user has an impression session where they can see several products at the same time. Each product can be clicked or not clicked by the user. The dataset consists of  1072226 rows and 55 columns. 

- features 
  - `user_id` is a unique identifier assigned to each user. This identifier has been anonymized and does not contain any information related to the real users. 
  - `product_id`  is a unique identifier assigned to each product, i.e. job offer. 
  - `impression_id` is a unique identifier assigned to each impression, i.e. online session that can have several products at the same time. 
  - `cat0` to `cat5` are anonymized categorical user features. 
  - `cat6` to `cat12` are anonymized categorical product features. 
  - `num13` to `num47` are anonymized numerical user features. 
- labels
  - `protected_attribute` is a binary feature that describes user gender proxy, i.e. female is 0, male is 1. The detailed description on the meaning can be found below. 
  - `senior` is a binary feature that describes the seniority of the job position, i.e. an assistant role is 0, a managerial role is 1. This feature was created during data processing step from the product title feature: if the product title contains words describing managerial role (e.g. 'president', 'ceo', and others), it is assigned to 1, otherwise to 0.  
  - `rank` is a numerical feature that corresponds to the positional rank of the product on the display for given `impression_id`. Usually, the position on the display creates the bias with respect to the click: lower rank means higher position of the product on the display. 
  - `displayrandom` is a binary feature that equals 1 if the display position on the banner of the products associated with the same `impression_id` was randomized. The click-rank metric should be computed on `displayrandom` = 1 to avoid positional bias. 
  - `click` is a binary feature that equals 1 if the product `product_id` in the impression `impression_id` was clicked by the user `user_id`. 


### Data statistics

| dimension           | average |
|---------------------|---------|
| click               | 0.077   |
| protected attribute | 0.500   |
| senior              | 0.704   |

### Protected attribute

As Criteo does not have access to user demographics we report a proxy of gender as protected attribute. 
This proxy is reported as binary for simplicity yet we acknowledge gender is not necessarily binary.

The value of the proxy is computed as the majority of gender attributes of products seen in the user timeline.
Product having a gender attribute are typically fashion and clothing. 
We acknowledge that this proxy does not necessarily represent how users relate to a given gender yet we believe it to be a realistic approximation for research purposes.

We encourage research in Fairness defined with respect to other attributes as well.


###  Limitations and interpretations 

We remark that the proposed gender proxy does not give a definition of the gender. Since we do not have access to the sensitive information, this is the best solution we have identified at this stage to idenitify bias on pseudonymised data, and we encourage any discussion on better approximations. This proxy is reported as binary for simplicity yet we acknowledge gender is not necessarily binary. Although our research focuses on gender, this should not diminish the importance of investigating other types of algorithmic discrimination. While this dataset provides important application of fairness-aware algorithms in a high-risk domain, there are several fundamental limitation that can not be addressed easily through data collection or curation processes. These limitations include historical bias that affect a positive outcome for a given user, as well as the impossibility to verify how close the gender-proxy is to the real gender value. Additionally, there might be bias due to the market unfairness. Such limitations and possible ethical concerns about the task should be taken into account while drawing conclusions from the research using this dataset. Readers should not interpret summary statistics of this dataset as ground truth but rather as characteristics of the dataset only. 

## Metrics

We strongly recommend to measure prediction quality using Negative Log-likelihood (lower is better).

We recommend to measure Fairness of ads by Demographic Parity conditioned on Senior job offers:

$$ E[f(x) | protected\_attribute=1, senior=1] - E[f(x) | protected\_attribute=0, senior=1] $$

This corresponds to the average difference in predictions for senior job opportunities between the two gender groups (lower is better).
Intuitively, when this metric is low it means we are not biased towards presenting more senior job opportunities (e.g. Director of XXX) to one gender vs the other.

## Example

You can start by running the example in `example.py` (requires numpy + torch). 
This implements 
- a dummy classifier (totally fair yet not very useful)
- a logistic regression with embeddings for categorical features (largely unfair and useful)
- a "fair" logistic regression (relatively fair and useful)

The "fair" logistic regression is based on the method proposed by [Bechavod & Ligett](https://arxiv.org/abs/1707.00044).

## Citation

If you use the dataset in your research please cite it using the following Bibtex excerpt:

```
@misc{criteo_fairness_dataset
author = {CRITEO},
title = {{CRITEO FAIRNESS IN JOB ADS DATASET},
year = {2024},
howpublished= {\url{http://XXX}}
```