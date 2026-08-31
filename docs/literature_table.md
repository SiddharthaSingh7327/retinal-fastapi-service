# Focused Literature Table


| Paper | Task | Uncertainty method | Routing action | Severity handling | Datasets | Main limitation |
|---|---|---|---|---|---|---|
| Jaskari et al. (2022), IEEE Access | Referable DR classification | Deep ensembles and test-time augmentation | Binary: refer or not, based on uncertainty | None. Treats referral as one class, no danger weighting | Private Finnish set + Kaggle EyePACS | Only accept/defer. No separate re-image action, no severity-aware priority |
| Band et al. (2021), NeurIPS D&B | Referable DR detection benchmark | Compares many Bayesian methods (MC Dropout, ensembles, etc.) | Selective prediction: defer the least certain cases | None. Binary referable target | EyePACS, plus a country-shift set (APTOS) | A benchmark, not a routing method. Binary, no image-quality or severity split |
| Ayhan et al. (2020), Medical Image Analysis | DR detection | Test-time data augmentation, expert-validated | Refer uncertain cases | None | Kaggle EyePACS + Messidor | Uncertainty-to-referral only. No four-way action, no harm cost |
| Mozannar and Sontag (2020), ICML | General learning-to-defer (not DR) | Learned defer score via a consistent surrogate loss | Defer to a single expert | None | General ML datasets, not fundus images | Single expert, one accept/defer choice, no medical severity or image quality |
| Fu et al. (2019), MICCAI | Retinal image quality assessment | Not uncertainty. A quality classifier | Reject or re-image poor photos | N/A (quality only) | EyeQ (Good/Usable/Reject), derived from EyePACS | Handles image quality alone. No diagnostic risk or routing to review |
| Liu et al. (2022), DeepDRiD | DR grading + image quality estimation challenge | None | None. A grading/quality challenge | Ordinal 0-4 grading | DeepDRiD (dual-view fundus) | A dataset/challenge, not a deferral system. No risk or action layer |

## Gap statement

Prior systems each solve only one part of the problem. Uncertainty-aware DR systems (Jaskari, Band, Ayhan) and learning-to-defer methods (Mozannar 
and Sontag) make only a single accept-or-defer choice based on how unsure the model is, with no sense of how dangerous an error would be, so a 
confident but severe miss is never prioritized. Image-quality work (Fu, Liu) handles re-imaging on its own, disconnected from diagnostic risk. 
The gap is that no existing system separates a bad-image failure from diagnostic risk and routes by clinical harm in one layer, so that likely 
under-graded severe disease is escalated even when the model is confident.