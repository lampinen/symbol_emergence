import tensorflow as tf
import numpy
import os
from itertools import combinations_with_replacement

######Parameters###################
#general
group_order = 14

#learning
init_eta = 0.0005
nepochs = 500000
early_termination_loss = 0.001
early_termination_max_dev = 0.2


curriculum_stage_0 = group_order+group_order*group_order+group_order*group_order*group_order 
curriculum_switch_epoch = 30000

RNN_seq_length = 4
embedding_size = group_order
nhidden_shared = group_order
nhidden_recurrent = embedding_size

grad_clip_norm = 5.

num_runs = 200
#rseed = 2  #reproducibility
###################################


identity = numpy.zeros(group_order)
identity[0] = 1.
elements = [numpy.roll(identity,i) for i in xrange(group_order)] 
empty_element = numpy.zeros(group_order)

def combine(a,b): 
    """group operation"""
    i = numpy.argmax(a)
    j = numpy.argmax(b)
    shift = (i+j) % group_order
    return numpy.roll(identity,shift) 

def combine_list(l):
    """applies group operation to a set of elements in a list l2r, producing outputs at each step, and ignoring empty_element defined above""" 
    if len(l) < 2:
	return l
    curr = l[0]
    outputs = [curr]
    for x in l[1:]:
	if numpy.array_equal(x,empty_element):
	    outputs.append(outputs[-1])
	else:
	    curr = combine(curr,x)
	    outputs.append(curr)
    return outputs

def pad(l,target_length,pad_element):
    """Pads list l to target length if not already there"""
    return l+(target_length-len(l))*[pad_element]

raw_x_data = []
masks = []
for nelements in xrange(1,RNN_seq_length+1):
    this_mask = [True]*nelements+[False]*(RNN_seq_length-nelements)
    for c in combinations_with_replacement(elements,nelements):
	raw_x_data.append(pad(list(c),RNN_seq_length,empty_element))
	masks.append(this_mask)
y_data = numpy.array([combine_list(x) for x in raw_x_data])
x_data = numpy.array(map(lambda l: map(lambda x: numpy.argmax(x),l),raw_x_data),dtype=numpy.int32)
masks = numpy.array(masks,dtype=numpy.bool)

for rseed in xrange(num_runs):
    print "run %i" %rseed
    filename_prefix = "results/cyclic/order_%i_seq_length-%i_nhidden-shared_%i-recurrent_%i-embedding_%i_rseed_%i_" %(group_order,RNN_seq_length,nhidden_shared,nhidden_recurrent,embedding_size,rseed)
#    if os.path.exists(filename_prefix+"final_a_reps.csv"):
#	print "skipping %i" %rseed
#	continue

    numpy.random.seed(rseed)
    tf.set_random_seed(rseed)

    input_ph = tf.placeholder(tf.int32, shape=[None,RNN_seq_length])
    target_ph =  tf.placeholder(tf.float32, shape=[None,RNN_seq_length,group_order])
    mask_ph = tf.placeholder(tf.bool, shape=[None,RNN_seq_length])

    embeddings = tf.Variable(tf.random_uniform([group_order,embedding_size],-0.1/embedding_size,0.1/embedding_size))

    embedded_inputs = tf.nn.embedding_lookup(embeddings,input_ph)
    
    Wr2h = tf.Variable(tf.random_normal([nhidden_recurrent,nhidden_shared],0.,1/numpy.sqrt(nhidden_recurrent)))
    We2h =  tf.Variable(tf.random_normal([embedding_size,nhidden_shared],0.,1/numpy.sqrt(embedding_size)))   
    bh = tf.Variable(tf.zeros([nhidden_shared,]))
    Wh2r = tf.Variable(tf.random_normal([nhidden_shared,nhidden_recurrent],0.,1/numpy.sqrt(nhidden_shared))) 
    br = tf.Variable(tf.zeros([nhidden_recurrent,]))

    output_logits = []
    recurrent_state = tf.squeeze(tf.slice(embedded_inputs,[0,0,0],[-1,1,-1]),axis=1)
    output_logits.append(tf.matmul(recurrent_state,tf.transpose(embeddings)))
    for i in xrange(1,RNN_seq_length):
	this_embedded_input = tf.squeeze(tf.slice(embedded_inputs,[0,i,0,],[-1,1,-1]),axis=1)
	hidden_state = tf.nn.relu(tf.matmul(recurrent_state,Wr2h)+tf.matmul(this_embedded_input,We2h)+bh)
	recurrent_state = tf.nn.tanh(tf.matmul(hidden_state,Wh2r)+br) 
	output_logits.append(tf.matmul(recurrent_state,tf.transpose(embeddings)))

    pre_outputs = tf.stack(output_logits,axis=1)
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=tf.boolean_mask(pre_outputs,mask_ph),labels=tf.boolean_mask(target_ph,mask_ph)))	

    output = tf.nn.softmax(pre_outputs)
    max_deviation = tf.reduce_max(tf.abs(tf.boolean_mask(output,mask_ph)-tf.boolean_mask(target_ph,mask_ph)))

    eta_ph = tf.placeholder(tf.float32)
    optimizer = tf.train.GradientDescentOptimizer(eta_ph)

    grads_and_vars = optimizer.compute_gradients(loss)
    grads = [x[0] for x in grads_and_vars]
    gvars = [x[1] for x in grads_and_vars]
    grads,__ = tf.clip_by_global_norm(grads,grad_clip_norm)
    grads_and_vars = zip(grads,gvars)
    train = optimizer.apply_gradients(grads_and_vars)

    #train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    def test_accuracy(x_data=x_data,y_data=y_data,masks=masks):
	curr_loss,curr_max_dev = sess.run([loss,max_deviation],feed_dict={input_ph: x_data,target_ph: y_data,mask_ph: masks})
	curr_loss /= len(x_data) 
	return curr_loss,curr_max_dev

#    def save_activations(tf_object,filename,remove_old=True):
#	if remove_old and os.path.exists(filename):
#	    os.remove(filename)
#	with open(filename,'ab') as fout:
#	    for i in xrange(len(a_data)):
#		numpy.savetxt(fout,sess.run(tf_object,feed_dict={a_ph: a_data[i].reshape([1,group_order]),b_ph: b_data[i].reshape([1,group_order])}).reshape((1,-1)),delimiter=',')
#
    def save_weights(tf_object,filename,remove_old=True):
	if remove_old and os.path.exists(filename):
	    os.remove(filename)
	with open(filename,'ab') as fout:
	    numpy.savetxt(fout,sess.run(tf_object),delimiter=',')


    def batch_train_with_standard_loss(x_data=x_data,y_data=y_data,masks=masks):
	sess.run(train,feed_dict={eta_ph: curr_eta,input_ph: x_data,target_ph: y_data,mask_ph: masks})

    print "Initial loss: %f, initial max deviation: %f" %(test_accuracy())

    curr_eta = init_eta
    filename = filename_prefix + "rep_tracks.csv"
    for epoch in xrange(nepochs):
	if epoch < curriculum_switch_epoch and epoch % 10 != 0:
	    batch_train_with_standard_loss(x_data=x_data[:curriculum_stage_0],y_data=y_data[:curriculum_stage_0],masks=masks[:curriculum_stage_0])
	else:
	    batch_train_with_standard_loss(x_data=x_data,y_data=y_data,masks=masks)

	if epoch % 1000 == 0:
	    curr_loss,curr_max_dev = test_accuracy(x_data=x_data,y_data=y_data,masks=masks)
	    print "epoch: %i, loss: %f, max_dev: %f" %(epoch, curr_loss, curr_max_dev)	
	    if curr_loss < early_termination_loss and curr_max_dev < early_termination_max_dev:
		print "Stopping early!"
		break 

    final_loss,final_max_dev = test_accuracy()
    print "Final loss: %f, maximum deviation: %f" %(final_loss,final_max_dev)

    if final_loss < 0.05 and final_max_dev < 0.2:
	save_weights(embeddings,filename_prefix+"final_reps.csv")
