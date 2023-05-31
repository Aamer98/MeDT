## In response to the concerns on the readability,  we provide a list of the changes and corresponding revised text

<ul>
  <li>R2: We now denote the components of Eq 1 and make algorithm 1 more descriptive to improve readability: [Algorithm 1](algorithm.png)</li>
  <li>R2: We move Section 3.1 on model training to Section 2 to improve the flow of the article.</li>
  <li>R3: We elucidate more clearly the difference between the baselines and the proposed method: “Behaviour cloning (BC) refers to a simple transformer that takes as input the past states and actions, guided by cross-entropy loss on predicted actions to directly imitate the behaviour of the clinician’s policy. The Decision Transformer (DT) builds on behaviour cloning by conditioning each trajectory with returns-to-go, which is the sum of future rewards. During inference, the RTG is set to the desired value (+1) to generate favourable actions.  The proposed method differs in that it additionally conditions on acuity-to-go at every time step. This helps to alleviate the burden of sparse rewards resulting from solely relying on RTG. Moreover, our modified SAPS2 score condition enables better interaction with the user by allowing for more fine-grained condition inputs to guide the generation of actions.”</li>
  <li>R3, R4: We make the captions of Fig 1 and 2 in supplementary more descriptive: [Fig 1](MeDT_train_v2.png) [Fig 2](MeDT_ev_v2.png)</li>
  <li>R4: We add descriptions for medical terms such as acuity score: “The acuity score provides an indication of the severity of illness of the patient in the ICU. Higher acuity scores attribute to higher illness severity.”</li>  
</ul>

## References

[1] Waechter, Jason et al. “Interaction Between Fluids and Vasoactive Agents on Mortality in Septic Shock: A Multicenter, Observational Study*.” Critical Care Medicine 42 (2014): 2158–2168. //

[2] Killian, Taylor W. et al. “An Empirical Study of Representation Learning for Reinforcement Learning in Healthcare.” ML4H@NeurIPS (2020). //

[3] Morkar, Dnyanesh N et al. “Comparative Study of Sofa, Apache Ii, Saps Ii, as a Predictor of Mortality in Patients of Sepsis Admitted in Medical ICU.” The Journal of the Association of Physicians of India vol. 70,4 (2022): 11-12.//

[4] Macdonald, S., Peake, S. L., Corfield, A. R., & Delaney, A. (2022). Fluids or vasopressors for the initial resuscitation of septic shock. Frontiers in medicine, 9, 1069782. https://doi.org/10.3389/fmed.2022.1069782