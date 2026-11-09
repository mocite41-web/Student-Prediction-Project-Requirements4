# SAD Mini-Analysis

**System:** Student Performance Prediction System

- **Problem:** Academic risk may only be noticed after poor results. This prototype explores early identification using available student indicators.
- **Users:** Faculty member or academic analyst.
- **Inputs:** Student attributes and academic indicators such as study time, absences, failures, G1, and G2.
- **Process:** Validate data → preprocess → apply trained model → classify risk.
- **Outputs:** AT RISK / NOT AT RISK label and model evaluation summary.
- **Key limitation:** A model trained on one Portuguese secondary-school dataset may not generalize to another institution or population.

## Reflection answers

1. **What is the prediction target, and why is G3 excluded from the input features?**  
   The target is whether the student is at risk based on the final Mathematics grade: G3 below 10 is AT RISK, while G3 of 10 or higher is NOT AT RISK. G3 is excluded from the inputs because it is the value the model is trying to predict.

2. **What does the model's recall for the 'At risk' class tell you?**  
   It tells us what portion of the actual at-risk students were correctly identified by the model.

3. **Why might a model trained on this dataset perform differently on students from your institution?**  
   The dataset came from Portuguese secondary schools. Students in another institution may have different backgrounds, learning conditions, grading practices, and academic patterns.

4. **What action should a faculty member take after seeing an 'AT RISK' prediction?**  
   The faculty member should review the student's situation, verify the information, and consider offering guidance or academic support. The prediction should not be the only basis for a decision.
