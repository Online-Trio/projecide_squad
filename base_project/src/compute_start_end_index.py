from hyper_param import *


def compute_start_end_index(training_df, validation_df, test_df):
    """
    For each dataframe adds two columns containing integers which indicate the initial and the final word of the answer
    inside the context.
    """
    print("Training set")
    training_df['start_index'] = 0
    training_df['end_index'] = 0
    print("Start computing indexes...")
    for i, row in training_df.iterrows():
        answer = row.text
        context = row.context
        start = len(context[:context.find(answer)].split(' ')) - 1  # count the words before the match
        end = start - 1 + len(answer.split(' '))
        training_df.loc[i, 'start_index'] = start
        training_df.loc[i, 'end_index'] = end
        if start == -1:
            print(i)    # check if, for any reason, context does not contain the answer
    print("Indexes computed")

    print("")
    print("Validation set")
    validation_df['start_index'] = 0
    validation_df['end_index'] = 0
    print("Start computing indexes...")
    for i, row in validation_df.iterrows():
        answer = row.text
        context = row.context
        start = len(context[:context.find(answer)].split(' ')) - 1
        end = start - 1 + len(answer.split(' '))
        validation_df.loc[i, 'start_index'] = start
        validation_df.loc[i, 'end_index'] = end
        if start == -1:
            print(i)    # check if, for any reason, context does not contain the answer
    print("Indexes computed")

    print("")
    print("Test set")
    test_df['start_index'] = 0
    test_df['end_index'] = 0
    print("Start computing indexes...")
    for i, row in test_df.iterrows():
        answer = row.text
        context = row.context
        start = len(context[:context.find(answer)].split(' ')) - 1
        end = start - 1 + len(answer.split(' '))
        test_df.loc[i, 'start_index'] = start
        test_df.loc[i, 'end_index'] = end
        if start == -1:
            print(i)    # check if, for any reason, context does not contain the answer
    print("Indexes computed")

    training_df.drop(columns=['answer_start', 'text'], inplace=True)
    validation_df.drop(columns=['answer_start', 'text'], inplace=True)
    test_df.drop(columns=['answer_start', 'text'], inplace=True)

    return training_df, validation_df, test_df


