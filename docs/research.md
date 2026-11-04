# Research papers

These 17 papers come from the research doc
[lastfirstdate: Matching Algorithm Research](https://claude.ai/artifact/Ky5sZrFrAbrUPyYsm8Kn68)
(31 Oct 2026). Each requirement in `docs/SPEC.md` names the papers it draws on. The links
below are the ones from that doc.

Papers 9 and 14 link to press releases about the paper rather than the paper itself. Read
the original paper before relying on a number from either.

## Papers by stage

| # | Paper | Stage | Requirements | Sprint | What we take from it |
|---|---|---|---|---|---|
| 1 | [Active Preference Learning with Discrete Choice Data](https://proceedings.nips.cc/paper_files/paper/2007/file/b6a1085a27ab7bff7550f8a3bd017df8-Paper.pdf), Brochu, de Freitas, Ghosh, NeurIPS 2007 | 2 | FR-LM-16, FR-LM-32 | 2, 6 | Show next whatever teaches the model most, so fewer swipes are needed |
| 2 | [Determinantal Point Processes for Machine Learning](https://arxiv.org/abs/1207.6083), Kulesza, Taskar, 2012 | 2 | FR-LM-16 | 2 | Pick a set of profiles that are as different from each other as possible |
| 3 | [Preference Learning with Gaussian Processes](https://mlanthology.org/icml/2005/chu2005icml-preference), Chu, Ghahramani, ICML 2005 | 2, 3 | FR-LM-15, FR-LM-16, FR-LM-32 | 2, 6 | Learn a taste function from likes and passes, with a measure of how sure it is |
| 4 | [Personalized Image Aesthetics](https://openaccess.thecvf.com/content_iccv_2017/html/Ren_Personalized_Image_Aesthetics_ICCV_2017_paper.html), Ren et al., ICCV 2017 | 3 | FR-LM-13, FR-LM-14, FR-LM-15 | 2 | General model plus a small personal correction, so a few ratings are enough |
| 5 | [Some Like It Hot: Visual Guidance for Preference Prediction](https://openaccess.thecvf.com/content_cvpr_2016/html/Rothe_Some_Like_It_CVPR_2016_paper.html), Rothe, Timofte, Van Gool, CVPR 2016 | 3 | FR-LM-13, FR-LM-14, FR-LM-15 | 2 | Predict one specific person's attraction to a face from the image |
| 6 | [ImRec: Learning Reciprocal Preferences Using Images](https://research-information.bris.ac.uk/en/publications/imrec-learning-reciprocal-preferences-using-images/), Neve, McConville, RecSys 2020 | 3, 5 | FR-LM-13, FR-LM-20 | 2, 3 | Photo preferences on both sides merged into one mutual score |
| 7 | [Photos Are All You Need for Reciprocal Recommendation in Online Dating](https://arxiv.org/abs/2108.11714), Neve, McConville, 2021 | 3 | FR-LM-13 | 2 | Photo history alone predicts mutual matches well |
| 8 | [Sentence-BERT](https://arxiv.org/abs/1908.10084), Reimers, Gurevych, 2019 | 4 | FR-LM-05, FR-LM-31 | 1 (interface), 6 (offline word stand-in), later (model) | Turn free-text interests into vectors so similar meanings score as similar |
| 9 | [Is Romantic Desire Predictable?](https://www.psychologicalscience.org/news/releases/romantic-matches-are-hard-to-predict.html), Joel, Eastwick, Finkel, Psychological Science 2017 | 4 | FR-LM-08, FR-LM-26 | 1, 5 | Traits measured before meeting barely predict desire for one specific person, so treat the personality weights as guesses and retune them from outcomes |
| 10 | [Reciprocal Recommender System for Online Dating](https://bibtex.github.io/RecSys-2010-PizzatoRCKYK.html), Pizzato et al., RecSys 2010 | 5 | FR-LM-20 | 3 | Score both directions, then combine them |
| 11 | [Fair Reciprocal Recommendation in Matching Markets](https://arxiv.org/abs/2409.00720v1), Tomita, Yokoyama, 2024 | 6 | FR-LM-22 | 4 | Nash social welfare: spread chances across everyone while keeping matches high |
| 12 | [Fairness of Exposure in Rankings](https://arxiv.org/abs/1802.07281v2), Singh, Joachims, KDD 2018 | 6 | FR-LM-21, FR-LM-23 | 4 | Rank for relevance while guaranteeing each profile a fair share of being seen |
| 13 | [Optimizing Rankings for Recommendation in Matching Markets](https://arxiv.org/abs/2106.01941v1), Su, Bayoumi, Joachims, WWW 2022 | 6 | FR-LM-22 | 4 | Optimize everyone's lists together, counting both sides and capacity limits |
| 14 | [Improving Match Rates in Dating Markets Through Assortment Optimization](https://www.informs.org/News-Room/INFORMS-Releases/News-Releases/Swipe-Left-or-Swipe-Right-New-Algorithm-Increases-Successful-Dating-Site-Matches-by-More-Than-252), Rios, Saban, Zheng, MSOM 2023 | 6 | FR-LM-24 | 4 | Hold back likely matches from members who just matched a lot |
| 15 | [Facilitating the Search for Partners on Matching Platforms](https://econpapers.repec.org/RePEc:inm:ormnsc:v:67:y:2021:i:10:p:5990-6029), Kanoria, Saban, Management Science 2021 | 6, 7 | FR-LM-09, FR-LM-10 (decision D1), FR-LM-34 | 1, 6 | Letting only the short side of the market initiate can raise welfare |
| 16 | [Managing Congestion in Matching Markets](https://econpapers.repec.org/RePEc:inm:ormsom:v:23:y:2021:i:3:p:620-636), Arnosti, Johari, Kanoria, MSOM 2021 | 6, 7 | FR-LM-09, FR-LM-21 (decisions D3, D4), FR-LM-34 | 1, 4, 6 | Capping applications cuts wasted effort on both sides |
| 17 | [Competing Bandits in Matching Markets](https://proceedings.mlr.press/v108/liu20c.html), Liu, Mania, Jordan, AISTATS 2020 | 8 | FR-LM-27 | 5 | Keep learning from noisy outcomes in a two-sided market by exploring a little |

## Reading order by sprint

- **Sprint 0:** no paper method is built yet. The synthetic population's split between
  shared and personal taste follows the idea in paper 9.
- **Sprint 1:** papers 15 and 16 (why she chooses and why caps help), paper 9 (why the
  personality weights are only a guess), paper 8 (what will replace interest tags later).
- **Sprint 2:** papers 4 and 5 (personal taste on top of a general model), then 3, then 1
  and 2 (choosing the 50), then 6 and 7.
- **Sprint 3:** papers 10 and 6 (making the score mutual).
- **Sprint 4:** papers 12, 11, 13, then 14.
- **Sprint 5:** paper 17.
