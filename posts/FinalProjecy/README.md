# MedicalBiasJLS
Final project for ML

Abstract
We aim to address the relationship between disease and procedures and sex and race in healthcare and how it reflects environmental inequities. Our approach is to train a model to generate risk likelihood scores and evaluate the relationship between different diseases and characteristics indicative of societal inequalities. We will then analyze the implications of these risk factors for fairness in the healthcare system. Our project will be successful if we are able to generate meaningful risk likelihood scores and have meaningful evaluation of their fairness.

Motivation and Question
We have been inspired by the Bias Replication Study that we discussed during lecture. We hope to look beyond healthcare operations to see actual societal implications. We are excited to use data that represents what is happening in our healthcare systems in real time. Algorithmic bias is incredibly problematic, and can further bias in the real world. Our machine learning model will allow us to reflect on bias in our healthcare and help us analyze the fairness in the system.

We have a synthetic data set, developed for an Introduction to Biomedical Science Textbook, for which we can develop predictive and exploratory models that would help us examine relationships between disease diagnoses and larger scale trends and impacts. 

Planned Deliverables
Our deliverables would include multiple aspects. A python package containing the development of our risk-assessment model. A Jupyter notebook consisting of initial visualizations of data and illustrating the use of our developed Python package to analyze our synthetic medical data. A written component would contain our analyses of our model’s risk score outputs, any biases reflected in our model, an evaluation of the relationship between different diseases and characteristics indicative of societal inequalities as well as a discussion of the 3 views of fairness as it pertains to our project. 

Partial success in our project would mean creating a risk-assessment model without analyzing the implications of our findings. To be fully successful in our final project we will need to fully assess multiple options to find the most optimal algorithm to classify our data. Secondly, we will analyze the fairness of our results in the context of existing societal inequities and biases.
Resources Required
Our project utilizes a synthetic data set created for an Introduction to Biomedical Data Science Textbook. The data was created using Synthea, a synthetic patient generator that models the medical history of synthetic patients. Synthea’s mission is “to output high-quality synthetic, realistic but not real, patient data and associated health records covering every aspect of healthcare.” This is also much easier to access than real patient data. 
Link to the Data: https://data.world/siyeh/synthetic-medical-data 


What You Will Learn

Julia: Looking back to my goal-setting reflection created at the beginning of the course, I wrote that I would like to complete all project milestones on time to keep myself and the group accountable in our semester time frame, as well as contribute to the project in an equal manner including in the algorithm implementation and project presentation, doing what I can to be a great group project partner. I am excited about this project proposal as it seems like we are all equally interested and invested in the topic, which sets us up for team-work success. I think I will be able to learn a lot through working on this project, namely to understand how algorithms can reveal biases and trends in the health of different populations. I think the thorough dissection of how our model and data pertains to the three views of fairness will help me learn how to evaluate fairness as a whole, as well as strengthen my evaluative writing skills. 

Lindsey: In my goal-setting reflection, I described my goal to conduct a series of experiments to assess algorithm bias in a certain class of algorithm, specifically an algorithm with broad real life application. This project proposal is similar, but not identical to this aspiration. This project also aims to analyze bias in terms of data with real life applications. However, instead of assessing a machine learning model, I am utilizing a machine learning model to understand the bias of a data set. Through this slightly altered project goal, I will be able to improve my ability to analyze machine learning models critically and become a socially responsible engineer which was my ultimate goal, not only for the project, but for the class as a whole.

Sophie: In my original goal-setting reflection, I wrote that I wanted to explore a project at the intersection of healthcare and bias. I think this proposal aligns perfectly with those goals, and I’m excited about the opportunity to explore a medical dataset deeply and better understand how algorithms can help enlighten biases in populations’ health. I am also looking forward to the opportunity to include aspects of scientific writing like figures and discussions of results, as that was another aspect of my reflection which I noted as skills I wanted to make use of/practice using.

Risk Statement
Risks that could potentially stop us from achieving our full deliverable would be if there are gaps in our chosen data that exclude identity groups. Additionally, if the data does not represent the real bias present in our healthcare system, we would have to analyze the lack of bias instead of the presence of it. While this would change the nature of what we anticipate to discover, we would still be able to complete this project. Ideally, in order to fully analyze the bias in our healthcare system we need data that accurately reflects the reality for different groups of people.


Ethics Statement
Our project has the potential to benefit groups that are marginalized by the healthcare system. By analyzing the fairness of the model we create based on healthcare data, we aim to highlight the historical and current inequities in the healthcare system. 
If our project does not identify biases reflected in the dataset, we could potentially harm the exact groups we are trying to help benefit, the groups that are marginalized by the healthcare system. Another potential harm is if the synthetic data contains gaps in our chosen data that exclude identity groups, we could potentially be harming those by not addressing them. 
The main purpose of our project is to explore exactly these kinds of potential algorithmic biases. While we won’t necessarily be mitigating these biases before they happen, as that would change our own analysis, the majority of our writing and visualizations will be spent picking apart these decisions for evidence of bias.

Tentative Timeline
Keeping in mind that we will have a checkpoint for the project in Week 9 or 10, and then final presentations in Week 12, our tentative timeline is as follows. Our goal for week three is to have our working risk-score algorithm and a few exploratory data analysis visualizations. By week six, we hope to have an analysis done for at least one selected condition/procedure, for example risk score for asthma by race and/or sex.

