# Audit: salman2024cnnlstmtf

Source: `data/papers/s00521-024-09558-5.pdf` (18 pages, 66,400 chars)

Salman, D., Direkoglu, C., Kusaf, M., Fahrioglu, M. (2024). "Hybrid deep
learning models for time series forecasting of solar power." *Neural
Computing and Applications* (Springer) 36:9095-9112.

## Coded fields

**year**: 2024 | "Neural Computing and Applications (2024) 36:9095-9112"

**venue**: Neural Computing and Applications (Springer) 36:9095-9112

**dataset**: Hourly data spanning four years, sourced from the Mendeley
Data repository (ref [32]); authors based in Northern Cyprus | "An input
size consisting of hourly data spanning four years was utilized in the
research project." (p.16); "Availability of data and materials: The used
data were taken from Mendeley database, as shown in reference [32]."
(p.20)

**night_hours_excluded = not_stated** | EXHAUSTIVE CHECK: zero matches
anywhere for night/daytime/daylight/sunrise/sunset/zenith/clear-sky/
diurnal in 66,400 characters.

**baseline_used = own_components** | "all of the possible combinations of
convolutional neural network (CNN), long short-term memory (LSTM), and
transformer (TF) models are experimented. These hybrid models also
compared with the single CNN, LSTM and TF models" (Abstract, p.1) - every
comparator is a combination or subset of the paper's own three building
blocks.

FLAG (verified NOT this paper's own baseline): the text does contain a
"persistence" mention - "The hybrid method proposed here is compared to
benchmark methods such as persistence, backpropagation neural network
(BPNN), and radial basis function neural network (RBFNN). The solar power
dataset used in this study was gathered from actual Limburg solar
farms." (p.5) - but this is Related Work, describing a DIFFERENT cited
study (which used Limburg, Netherlands solar farm data, not this paper's
own Mendeley/Northern-Cyprus-affiliated dataset). Read carefully to avoid
misattributing another paper's persistence baseline to this one.

**skill_score_reported = no** | EXHAUSTIVE CHECK: zero matches for "skill
score"/"forecast skill"/"relative to persistence"/"SS"/"improvement over
persistence" describing this paper's own results. Metrics are MAE, MSE,
RMSE (p.15-16).

**weather_source = not_stated** | The paper never explicitly states
whether its own model's input is univariate (PV power only) or includes
weather variables - unlike most other papers in this batch, no feature
list is given anywhere in the extracted Methodology/Experimental Setup
sections (pp.14-17). The "weather change features" language found in the
text (p.5) describes the DIFFERENT cited Limburg study noted above, not
confirmed as this paper's own input composition.

**split_type = not_stated** | "The data were passively split into three
groups: training (80%), validation (5%) and testing (15%)." (p.16) -
ratios given, no statement of chronological vs. random ordering.

**n_seeds = not_stated**

**variance_reported = no** | EXHAUSTIVE CHECK: zero matches anywhere for
seed/standard deviation/"+/-"/confidence interval/error bars/multiple
runs/repeated trials.

**code_available = not_stated** | Data availability names a specific
public repository (Mendeley Data, ref [32]) rather than an "on request"
gate - a genuine positive for DATA, recorded separately per the coding
rules - but the Declarations section says nothing about code.

**key_claim**: "For 24-hour-ahead solar power forecasting (four years of
hourly data from Mendeley Data, 80/5/15 train/validation/test split,
ordering not stated), the CNN-LSTM-TF hybrid model achieves the lowest
error (MAE=0.551% with the Nadam optimizer) among all tested combinations
of CNN/LSTM/Transformer, while the TF-LSTM combination performs
comparatively poorly (MAE=16.17%); results are also compared across five
different optimizers (Adam, Nadam, Adamax, RMSprop, AMSGrad)." | Abstract
(p.1); "The model was tasked with making predictions for the next
twenty-four hours." (p.16)

## What this paper does WELL

- Exhaustively tests EVERY combination of its three building blocks (CNN,
  LSTM, Transformer - 2^3-1 = 7 non-empty combinations implied) rather
  than only the full proposed stack versus one or two ablations - the most
  thorough component-attribution design seen in this batch.
- Additionally varies the OPTIMIZER (5 choices) as a second experimental
  axis crossed with model architecture, and reports that architecture
  choice interacts with optimizer choice (best/worst results both involve
  specific optimizer pairings) - a genuine two-factor sensitivity study,
  relevant in spirit to this project's own RQ2 protocol-sensitivity
  framing even though the varied factor here is a training
  hyperparameter, not an evaluation-construction choice.
- Explicit, single, stated forecast horizon (24 hours).
- Names a specific, checkable public data repository rather than an
  "available on request" statement.

## Other observations

- The paper's own worst-performing combination (TF-LSTM, MAE=16.17%) is
  roughly 30x worse than its best (CNN-LSTM-TF, MAE=0.551%) - an
  unusually large spread across architecture choices for what is
  presumably the same dataset and split, worth noting as an example of
  how much apparent "model capacity" differences can be architecture-
  ordering artifacts (CNN-LSTM-TF vs TF-LSTM differ only in stage order)
  rather than genuine capability differences - directly relevant to this
  project's own central thesis, though the paper itself does not frame it
  this way.
- No shared authors/institution with any other paper coded in this batch
  (Middle East Technical University Northern Cyprus / Cyprus International
  University).
