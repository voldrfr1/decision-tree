import pandas as pd

test_data= pd.DataFrame({
    'job_title': ['Data Scientist', 'Software Engineer', 'Data Scientist', 'Software Engineer', 'AI/ML Engineer'],
    'classification_tasks_frequency': ['very often', 'rarely', 'often', 'sometimes', 'often'],
    'motivation_to_learn': ['Low', 'Medium', 'Low', 'High', 'High'],
    'should_learn_dt': [1, 0, 0, 1, 1]  # 1 indicates 'Yes', 0 indicates 'No'
})

train_data = pd.DataFrame({
    'job_title': ['Data Scientist', 'Software Engineer', 'AI/ML Engineer', 'Data Scientist', 'Software Engineer',
                  'AI/ML Engineer', 'Data Scientist', 'Software Engineer', 'AI/ML Engineer'],
    'classification_tasks_frequency': ['often', 'rarely', 'sometimes', 'very often', 'often',
                                       'sometimes', 'very often', 'rarely', 'often'],
    'motivation_to_learn': ['High', 'Low', 'Medium', 'Low', 'High', 'Medium', 'High', 'Low', 'Medium'],
    'should_learn_dt': [1, 0, 1, 0, 1, 1, 0, 1, 0]  # Adjusted values for testing
})