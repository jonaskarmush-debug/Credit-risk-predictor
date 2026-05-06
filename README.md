Credit Risk Predictor
Built by Jonas Karmush


IMPORTANT NOTE BEFORE YOU START!
This is my absolute first code and i've genuinely no idea how to make this simpler but this is the only way ive found to start the app, I'll look for a way to make it easier in the near future
Step 1: Download the ZIP file and extract it somewhere on your computer
Step 2: Install required libraries: how you do this is start CMD on ur PC, type in this
py -m pip install pandas scikit-learn matplotlib joblib

Now you've got the libraries installed (You will never need to do this twice)
Now you're basically done! Open up credit_risk_full_program.py; press F5 and hit RUN, you should see ''Automation Complete'' once you've completed the task, and as same as before, you never have to do this twice
Now you're done; open up credit_risk_desktop_app_3_5.py, press F5, Run it and boom its gonna start! 
You're finished!

And if you're like me and really lazy i've also created a .Bat file that you'll find under ''run_full_application'' this .Bat file will work anytime after youve done these steps!


---

So what is this?

This is a desktop app that predicts whether someone is likely to be a good or bad credit risk. You plug in some info about a borrower — things like their age, loan amount, credit history, savings — and a machine learning model spits out a probability of default along with a recommendation on whether to approve or reject the loan.

It's built to simulate the kind of tool that banks and lenders actually use. Turns out logistic regression is still the go-to algorithm in real credit scoring because regulators require banks to be able to explain their decisions, and logistic regression is about as explainable as it gets.

---
 How to run it

You'll need Python and a few libraries. If you don't have them yet:

```
pip install pandas scikit-learn matplotlib joblib
```

Then, **run this first** to train the model (you only need to do this once):

```
py credit_risk_full_program.py
```

After that, launch the actual app:

```
py credit_risk_desktop_app_3.py
```

---

 What do all the fields mean?

 Left side — the numbers

| Field | What it is |
|---|---|
| Loan duration (months) | How long the loan runs. 1 to 120 months. |
| Loan amount (SEK) | How much they want to borrow. 1,000 to 1,000,000 SEK. |
| Current debt (SEK) | How much debt they already have before this loan. |
| Repayment burden score (1-4) | How much of their income goes to loan repayments. 1 is chill, 4 is stressful. |
| Years at current residence (1-4) | How long they've lived at their current address. This is a score from 1 to 4, not actual years — that's just how the original dataset encoded it. 1 = less than a year, 4 = 4+ years. |
| Age | Their age. |
| Existing active loans/credits | How many loans or credit lines they already have open. |
| Number of dependents | How many people rely on them financially. |

 Right side — the dropdowns

| Field | What it is |
|---|---|
| Housing | Own, rent, or living for free. |
| Checking account status | What their checking account looks like — empty, healthy, or no account at all. |
| Credit history | Have they paid things back before, or is there a history of delays? |
| Savings status | Roughly how much they have saved up. |
| Employment length | How long they've been at their current job. |
| Loan purpose | What the money is actually for. |
| Property / collateral | What assets they have that could back the loan. |
| Other payment plans | Do they have other active payment plans running? |

---

 Reading the results

 Risk levels

| Level | Probability | What it means |
|---|---|---|
| LOW RISK | 0–25% | Looks solid, probably fine |
| MEDIUM RISK | 25–50% | A few concerns, worth keeping an eye on |
| HIGH RISK | 50–75% | Real default risk here |
| VERY HIGH RISK | 75–100% | Very likely to default |

Recommendations

| Recommendation | When it shows up |
|---|---|
| APPROVE | Below 25% bad risk |
| APPROVE WITH CAUTION | 25–50% bad risk |
| REVIEW CAREFULLY | 50–75% bad risk |
| DECLINE | Above 75% bad risk |

 The tabs

Risk Summary** — shows up automatically after each prediction. Instead of just showing the result, it breaks down the specific risk factors for that borrower — things like long loan duration, weak savings, or a bad checking account status.

Borrower History** — keeps a running log of every prediction you make during the session so you can compare borrowers side by side. There's a clear button if you want to start fresh.

Save Prediction** — saves everything to a `predictions_log.csv` file in your project folder. Date, time, borrower details, probabilities, decision, the works.

---

 The model

I used **Logistic Regression** trained on the **German Credit Dataset** from OpenML — 1,000 real credit cases. The target variable is simple: good credit risk or bad credit risk.

### How it performs

| Metric | Score |
|---|---|
| Accuracy | 77.5% |
| ROC-AUC | 0.801 |

Not perfect, but honestly pretty solid for a logistic regression on a dataset this size. It's better at catching good borrowers (83% precision) than bad ones, which is a known tradeoff in credit scoring — you don't want to reject good customers, but you also don't want to approve everyone.

---

## A note on the residence field

The "Years at current residence" field only goes from 1 to 4. That's not a mistake — the original dataset encoded it as a category rather than actual years. The model has never seen values outside that range, so going beyond 4 wouldn't mean anything to it anyway.

---

## Project files

```
PROJEKT/
│
├── credit_risk_desktop_app_3.py   # The main app
├── credit_risk_full_program.py    # Trains and saves the model
├── credit_risk_app.py             # A Streamlit web version (bonus)
├── credit_risk_program.py         # Simple script version
│
├── models/
│   ├── credit_risk_model.pkl      # The trained model
│   └── model_columns.pkl          # Column names the model expects
│
├── outputs/
│   ├── model_metrics.csv
│   ├── classification_report.txt
│   ├── roc_curve.png
│   └── coefficients.csv
│
└── predictions_log.csv            # Created when you first save a prediction
```

---

*Built by Jonas Karmush, 2026*
