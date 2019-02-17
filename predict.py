import tensorflow as tf 
import tensorflow_hub as hub 
import tokenization
import pandas as pd
import numpy as np

MODULE_URL = "https://tfhub.dev/google/bert_cased_L-12_H-768_A-12/1"

def create_tokenizer_from_hub_module():
  """Get the vocab file and casing info from the Hub module."""
  with tf.Graph().as_default():
    bert_module = hub.Module(MODULE_URL)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    with tf.Session() as sess:
      vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                            tokenization_info["do_lower_case"]])
  return tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case)

def convert_single_text(text, max_seq_length, tokenizer):
    tokens_a = tokenizer.tokenize(text)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids

def align(text, tokens):
    j = 0
    alignments = [-1 for _ in range(len(text))]
    for i in range(len(tokens)):
        token = tokens[i]
        token = token.replace(" ##", "")
        token = token.replace("##", "")
        for c in token:
            while True:
                j = j + 1
                alignments[j-1] = i
                if text[j-1] == c:
                    break
    return alignments

def preprocess_label(df):
    labels = []
    for each in df.iterrows():
        d = each[1]
        A_coref = d["A-coref"]
        B_coref = d["B-coref"]
        if A_coref:
            labels.append(0)
        elif B_coref:
            labels.append(1)
        else:
            labels.append(2)
    return labels

def preprocess(df, max_seq_length, tokenizer):
    batch_input_ids = []
    batch_input_mask = []
    batch_segment_ids = []
    batch_p_mask = []
    batch_a_mask = []
    batch_b_mask = []
    for each in df.iterrows():
        d = each[1]
        text = d["Text"]
        P = d["Pronoun"]
        P_offset = d["Pronoun-offset"]
        A = d["A"]
        A_offset = d["A-offset"]
        B = d["B"]
        B_offset = d["B-offset"]
        tokens = tokenizer.tokenize(text)
        alignments = align(text, tokens)
        P_tok_offset = alignments[P_offset] + 1
        A_tok_offset = alignments[A_offset] + 1
        B_tok_offset = alignments[B_offset] + 1
        p_mask = [[1]*768 if i == P_tok_offset else [0]*768 for i in range(max_seq_length)]
        a_mask = [[1]*768 if i == A_tok_offset else [0]*768 for i in range(max_seq_length)]
        b_mask = [[1]*768 if i == B_tok_offset else [0]*768 for i in range(max_seq_length)]
        input_ids, input_mask, segment_ids = convert_single_text(text, max_seq_length, tokenizer)
        for batch, elem in zip([batch_input_ids, batch_input_mask, batch_segment_ids,batch_p_mask,batch_a_mask,batch_b_mask],
                                [input_ids, input_mask, segment_ids, p_mask, a_mask, b_mask]):
            batch.append(elem)
    return batch_input_ids, batch_input_mask, batch_segment_ids,batch_p_mask,batch_a_mask,batch_b_mask
       
def get_bert_embedding(batch_input_ids,batch_input_mask,batch_segment_ids,batch_p_mask,batch_a_mask,batch_b_mask):
    max_seq_length = len(batch_input_ids[0])
    batch_size = len(batch_input_ids)
    print(batch_size)
    # placeholders
    pf_input_ids = tf.placeholder(tf.int32,[None,max_seq_length])
    pf_input_mask = tf.placeholder(tf.int32,[None,max_seq_length])
    pf_segment_ids = tf.placeholder(tf.int32,[None,max_seq_length])
    pf_p_mask = tf.placeholder(tf.int32,[None,max_seq_length,768])
    pf_a_mask = tf.placeholder(tf.int32,[None,max_seq_length,768])
    pf_b_mask = tf.placeholder(tf.int32,[None,max_seq_length,768])
    # bert embeddings
    bert_module = hub.Module(MODULE_URL)
    bert_inputs = dict(
        input_ids=pf_input_ids,
        input_mask=pf_input_mask,
        segment_ids=pf_segment_ids)
    bert_outputs = bert_module(bert_inputs, signature="tokens", as_dict=True)
    #pooled_output = bert_outputs["pooled_output"]
    sequence_output = bert_outputs["sequence_output"]

    p_embedding = tf.reduce_sum(sequence_output * tf.cast(pf_p_mask,tf.float32), axis = 1)
    a_embedding = tf.reduce_sum(sequence_output * tf.cast(pf_a_mask,tf.float32), axis = 1)
    b_embedding = tf.reduce_sum(sequence_output * tf.cast(pf_b_mask,tf.float32), axis = 1)
    embedding = tf.concat([p_embedding, a_embedding, b_embedding], axis = 1)

    result = np.zeros([batch_size,768*3])
    mb_size = 8
    mb_length = batch_size // mb_size
    sess = tf.Session()
    sess.run(tf.initializers.global_variables())
    for mb_idx in range(mb_length):
        if mb_idx != mb_length - 1:
            start = mb_idx*mb_size
            end = (mb_idx+1)*mb_size
        else:
            start = mb_idx*mb_size
            end = batch_size
        print(mb_idx)
        feed_dict = {
            pf_input_ids: batch_input_ids[start:end],
            pf_input_mask: batch_input_mask[start:end],
            pf_segment_ids: batch_segment_ids[start:end],
            pf_p_mask: batch_p_mask[start:end],
            pf_a_mask: batch_a_mask[start:end],
            pf_b_mask: batch_b_mask[start:end]
        }
        result[start:end,:] = sess.run(embedding,feed_dict=feed_dict)
        
    return result

def create_X_y(df, max_seq_length, tokenizer, is_predict=False):
    batch_input_ids, batch_input_mask, batch_segment_ids,batch_p_mask,batch_a_mask,batch_b_mask = preprocess(df,max_seq_length,tokenizer)
    print('preprocess done')
    X = get_bert_embedding(batch_input_ids, batch_input_mask, batch_segment_ids,batch_p_mask,batch_a_mask,batch_b_mask)
    print('bert embeddding obtained')
    if is_predict:
        y = None
    else:
        y = preprocess_label(df)
    return X, y


def main():
    path = './data/'
    df_gap_test = pd.read_csv( path+'gap-test.tsv', delimiter='\t' )
    df_gap_dev = pd.read_csv( path+'gap-development.tsv', delimiter='\t' )
    df_gap_valid = pd.read_csv( path+'gap-validation.tsv', delimiter='\t' )
    df_kernel_test = pd.read_csv( path+'test_stage_1.tsv', delimiter='\t' )

    df_train = pd.concat([df_gap_dev,df_gap_test,df_gap_valid])
    df_test = df_kernel_test

    tokenizer = create_tokenizer_from_hub_module()
    X_test, _ = create_X_y(df_test, 128, tokenizer, is_predict = True)
    X_train, y_train = create_X_y(df_train, 128, tokenizer)


    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty="l2", dual=False, tol=0.00001, 
                                C=0.01,
                                class_weight='balanced',
                                fit_intercept=True, intercept_scaling=1,
                                random_state=43, solver="liblinear",
                                max_iter=100, multi_class="ovr", verbose=0,
                                warm_start=False, n_jobs=1)

    model.fit(X_train,y_train)
    print(model.score(X_train,y_train))

    proba = model.predict_proba(X_test)
    #labels = model.predict(X_test)

    kaggle_submission_df = pd.DataFrame({
        'ID': pd.Series(df_test['ID'].values),
        'A': pd.Series(proba[:,0]),
        'B': pd.Series(proba[:,1]),
        'NEITHER': pd.Series(proba[:,2]),
    })

    kaggle_submission_df.to_csv('./submission/kaggle_submission2.csv',index=False)

if __name__ == "__main__":
    main()



