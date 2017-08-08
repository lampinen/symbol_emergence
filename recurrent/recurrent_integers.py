import tensorflow as tf
import numpy
import os
from itertools import product 

######Parameters###################
#general
max_n = 20

#learning
init_eta = 0.0005
nepochs = 500000
early_termination_loss = 0.001
early_termination_max_dev = 0.2

RNN_seq_length = 4
embedding_size = max_n 
nhidden_shared = max_n
nhidden_recurrent = embedding_size

grad_clip_norm = 5.

num_runs = 200
#rseed = 2  #reproducibility
###################################
identity = numpy.zeros(max_n + 1)
identity[0] = 1.
numbers = [numpy.roll(identity,i) for i in range(max_n + 1)] 
empty_element = numpy.zeros(max_n + 1)

task_names = ["add", "subtract", "divide", "multiply"]

def multiply(a, b): 
    """group operation"""
    i = numpy.argmax(a)
    j = numpy.argmax(b)
    shift = i * j
    if shift > max_n:
        return None
    return numpy.roll(identity, shift) 

def add(a, b): 
    """group operation"""
    i = numpy.argmax(a)
    j = numpy.argmax(b)
    shift = i + j
    if shift > max_n:
        return None
    return numpy.roll(identity, shift) 

def divide(a, b): 
    """group operation"""
    i = numpy.argmax(a)
    j = numpy.argmax(b)
    if (j == 0) or (i % j != 0):
        return None
    shift = i // j
    return numpy.roll(identity, shift) 

def subtract(a, b): 
    """group operation"""
    i = numpy.argmax(a)
    j = numpy.argmax(b)
    if i < j:
        return None
    shift = i - j
    return numpy.roll(identity, shift) 

task_mappings = {"add": {"operation": add}, 
                 "subtract": {"operation": subtract},
                 "multiply": {"operation": multiply},
                 "divide": {"operation": divide}}

def combine_list(l, operation):
    """applies operation to a set of elements in a list l2r, producing outputs at each step, and ignoring empty_element defined above""" 
    if len(l) < 2:
        return l
    curr = l[0]
    outputs = [curr]
    for x in l[1:]:
        if numpy.array_equal(x,empty_element):
            outputs.append(outputs[-1])
        else:
            curr = operation(curr,x)
            outputs.append(curr)
    return outputs

def pad(l,target_length,pad_element):
    """Pads list l to target length if not already there"""
    return l+(target_length-len(l))*[pad_element]

def valid(l):
    """Returns false if list contains a None, True otherwise"""
    return all([x is not None for x in l])

def build_data_set(operation, RNN_seq_length=RNN_seq_length):
    """Constructs a dataset for the given operation function"""
    raw_x_data = []
    masks = []
    for nelements in range(1,RNN_seq_length+1):
        this_mask = [True]*nelements+[False]*(RNN_seq_length-nelements)
        for c in product(numbers,repeat=nelements):
            raw_x_data.append(pad(list(c),RNN_seq_length,empty_element))
            masks.append(this_mask)
    y_data = numpy.array([combine_list(x, operation) for x in raw_x_data])
    valid_points = list(map(valid,y_data))
    y_data = numpy.array(list(map(list,y_data[valid_points])))
    x_data = numpy.array([[numpy.argmax(x) for x in l] for l in numpy.array(raw_x_data)[valid_points]],dtype=numpy.int32)
    masks = numpy.array(numpy.array(masks,dtype=numpy.bool)[valid_points])
    return x_data, y_data, masks


for task_name in task_names:
    x_data, y_data, masks = build_data_set(task_mappings[task_name]["operation"])
    task_mappings[task_name].update({"x_data": x_data, "y_data": y_data, "masks": masks})

def build_operation_subnetwork(embedded_inputs, target_ph, mask_ph, task_name): 
    with tf.variable_scope(task_name):
        Wr2h = tf.Variable(tf.random_normal([nhidden_recurrent,nhidden_shared],0.,1/numpy.sqrt(nhidden_recurrent)))
        We2h =  tf.Variable(tf.random_normal([embedding_size,nhidden_shared],0.,1/numpy.sqrt(embedding_size)))   
        bh = tf.Variable(tf.zeros([nhidden_shared,]))
        Wh2r = tf.Variable(tf.random_normal([nhidden_shared,nhidden_recurrent],0.,1/numpy.sqrt(nhidden_shared))) 
        br = tf.Variable(tf.zeros([nhidden_recurrent,]))

        output_logits = []
        recurrent_state = tf.squeeze(tf.slice(embedded_inputs,[0,0,0],[-1,1,-1]),axis=1)
        output_logits.append(tf.matmul(recurrent_state,tf.transpose(embeddings)))

        for i in range(1,RNN_seq_length):
            this_embedded_input = tf.squeeze(tf.slice(embedded_inputs,[0,i,0,],[-1,1,-1]),axis=1)
            hidden_state = tf.nn.relu(tf.matmul(recurrent_state,Wr2h)+tf.matmul(this_embedded_input,We2h)+bh)
            recurrent_state = tf.nn.tanh(tf.matmul(hidden_state,Wh2r)+br) 
            output_logits.append(tf.matmul(recurrent_state,tf.transpose(embeddings)))

        pre_outputs = tf.stack(output_logits,axis=1)

        output = tf.nn.softmax(pre_outputs)
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=tf.boolean_mask(pre_outputs,mask_ph),labels=tf.boolean_mask(target_ph,mask_ph)))        
        max_deviation = tf.reduce_max(tf.abs(tf.boolean_mask(output,mask_ph)-tf.boolean_mask(target_ph,mask_ph)))
    return output, loss, max_deviation
    
def get_train_op(loss, optimizer):
    grads_and_vars = optimizer.compute_gradients(loss)
    grads = [x[0] for x in grads_and_vars]
    gvars = [x[1] for x in grads_and_vars]
    grads,__ = tf.clip_by_global_norm(grads,grad_clip_norm)
    grads_and_vars = list(zip(grads,gvars))
    train = optimizer.apply_gradients(grads_and_vars)
    return train


for rseed in range(num_runs):
    print("run %i" %rseed)
    filename_prefix = "results/integers/maxn_%i_seq_length-%i_nhidden-shared_%i-recurrent_%i-embedding_%i_rseed_%i_" %(max_n,RNN_seq_length,nhidden_shared,nhidden_recurrent,embedding_size,rseed)
#    if os.path.exists(filename_prefix+"final_a_reps.csv"):
#        print "skipping %i" %rseed
#        continue

    numpy.random.seed(rseed)
    tf.set_random_seed(rseed)

    input_ph = tf.placeholder(tf.int32, shape=[None,RNN_seq_length])
    target_ph =  tf.placeholder(tf.float32, shape=[None,RNN_seq_length,max_n])
    mask_ph = tf.placeholder(tf.bool, shape=[None,RNN_seq_length])

    embeddings = tf.Variable(tf.random_uniform([max_n+1,embedding_size], -0.1/embedding_size, 0.1/embedding_size))

    embedded_inputs = tf.nn.embedding_lookup(embeddings,input_ph)
    
    # Create networks.
    for task_name in task_names:
        output, loss, max_deviation = build_operation_subnetwork(embedded_inputs, target_ph, mask_ph, task_name=task_name) 
        task_mappings[task_name].update({"output": output, "loss": loss, "max_deviation": max_deviation})
    
    # Create training ops
    eta_ph = tf.placeholder(tf.float32)
    optimizer = tf.train.GradientDescentOptimizer(eta_ph)
    
    for task_name in task_names:
        task_mappings[task_name]["train"] = get_train_op(task_mappings[task_name]["loss"], optimizer)        

    # init
    init = tf.global_variables_initializer()

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    sess.run(init)

    def test_accuracy(name): 
        curr_loss, curr_max_dev = sess.run([task_mappings[name]["loss"], task_mappings[name]["max_deviation"]], feed_dict={input_ph: task_mappings[name]["x_data"], target_ph: task_mappings[name]["y_data"], mask_ph: task_mappings[name]["masks"]})
        curr_loss /= len(x_data) 
        return curr_loss, curr_max_dev

    def save_weights(tf_object,filename,remove_old=True):
        if remove_old and os.path.exists(filename):
            os.remove(filename)
        with open(filename,'ab') as fout:
            numpy.savetxt(fout,sess.run(tf_object),delimiter=',')

    def batch_train_with_standard_loss(name):
        sess.run(task_mappings[name]["train"],feed_dict={eta_ph: curr_eta, input_ph: task_mappings[name]["x_data"], target_ph: task_mappings[name]["y_data"], mask_ph: task_mappings[name]["masks"]})

    for task_name in task_names:
        curr_loss, curr_max_dev = test_accuracy(task_name)
        print("Task: %s, initial loss: %f, initial max deviation: %f" %(task_name, curr_loss, curr_max_dev))

    curr_eta = init_eta
    filename = filename_prefix + "rep_tracks.csv"
    for epoch in range(nepochs):
        for task_name in task_names:
            batch_train_with_standard_loss(task_name)

        if epoch % 1000 == 0:
            losses = []
            max_devs = []
            for task_name in task_names:
                curr_loss, curr_max_dev = test_accuracy(task_name)
                losses.append(curr_loss)
                max_devs.append(curr_max_dev)
                print("Task: %s, epoch: %i, loss: %f, initial max deviation: %f" %(task_name, epoch, curr_loss, curr_max_dev))
            
            if max(losses) < early_termination_loss and max(max_devs) < early_termination_max_dev:
                print("Stopping early!")
                break 

    losses = []
    max_devs = []
    for task_name in task_names:
        curr_loss, curr_max_dev = test_accuracy(task_name)
        losses.append(curr_loss)
        max_devs.append(curr_max_dev)
        curr_loss, curr_max_dev = test_accuracy(task_name)
        print("Task: %s, final loss: %f, final max deviation: %f" %(task_name, curr_loss, curr_max_dev))

    if max(losses) < 0.05 and max(max_devs) < 0.2:
        save_weights(embeddings, filename_prefix+"final_reps.csv")
