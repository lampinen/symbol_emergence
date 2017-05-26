import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
import os

######Parameters###################
#general
group_order = 6

#learning
init_eta = 0.0005
eta_decay = 1.0 #multiplicative per eta_decay_epoch epochs
eta_decay_epoch = 10
nepochs = 100000
early_termination_loss = 0.005
nhidden_separate = group_order
nhidden_shared = group_order

num_runs = 100
#rseed = 2  #reproducibility
###################################

identity = numpy.zeros(group_order)
identity[0] = 1.
elements = [numpy.roll(identity,i) for i in xrange(group_order)] 

def combine(a,b): 
    """group operation"""
    i = numpy.argmax(a)
    j = numpy.argmax(b)
    shift = (i+j) % group_order
    return numpy.roll(identity,shift) 

#print combine(elements[0],elements[1])
#print combine(elements[4],elements[5])
#print combine(elements[2],elements[2])

raw_x_data = [(a,b) for a in elements for b in elements]
y_data = numpy.array([combine(x[0],x[1]) for x in raw_x_data])
a_data = numpy.array([a for (a,b) in raw_x_data])
b_data = numpy.array([b for (a,b) in raw_x_data])

for rseed in xrange(num_runs):
    print "run %i" %rseed
    filename_prefix = "results/cyclic/order_%i_nhidden-shared_%i-separate_%i_rseed_%i_" %(group_order,nhidden_shared,nhidden_separate,rseed)
#    if os.path.exists(filename_prefix+"final_a_reps.csv"):
#	print "skipping %i" %rseed
#	continue

    numpy.random.seed(rseed)
    tf.set_random_seed(rseed)

    a_ph = tf.placeholder(tf.float32, shape=[None,group_order])
    b_ph = tf.placeholder(tf.float32, shape=[None,group_order])
    target_ph =  tf.placeholder(tf.float32, shape=[None,group_order])


    W1a = tf.Variable(tf.random_normal([group_order,nhidden_separate],0.,1/numpy.sqrt(nhidden_separate)))
    b1a = tf.Variable(tf.zeros([nhidden_separate,]))
    W1b = tf.Variable(tf.random_normal([group_order,nhidden_separate],0.,1/numpy.sqrt(nhidden_separate)))
    b1b = tf.Variable(tf.zeros([nhidden_separate,]))
    W2 = tf.Variable(tf.random_normal([2*nhidden_separate,nhidden_shared],0.,1/numpy.sqrt(nhidden_shared)))
    b2 = tf.Variable(tf.zeros([nhidden_shared,]))
    W3 = tf.Variable(tf.random_normal([nhidden_shared,group_order],0.,1/numpy.sqrt(group_order)))
    b3 = tf.Variable(tf.zeros([group_order,]))

    a_rep = tf.nn.relu(tf.matmul(a_ph,W1a)+b1a) 
    b_rep = tf.nn.relu(tf.matmul(b_ph,W1b)+b1b) 

    middle_rep = tf.nn.relu(tf.matmul(tf.concat([a_rep,b_rep],1),W2)+b2)
    pre_output = tf.matmul(middle_rep,W3)+b3
    output = tf.nn.softmax(pre_output)

    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=target_ph,logits=pre_output))
    eta_ph = tf.placeholder(tf.float32)
    optimizer = tf.train.GradientDescentOptimizer(eta_ph)
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    def test_accuracy():
	curr_loss = sess.run(loss,feed_dict={a_ph: a_data,b_ph: b_data,target_ph: y_data})
	curr_loss /= group_order*group_order
	return curr_loss

    def save_activations(tf_object,filename,remove_old=True):
	if remove_old and os.path.exists(filename):
	    os.remove(filename)
	with open(filename,'ab') as fout:
	    for i in xrange(len(a_data)):
		numpy.savetxt(fout,sess.run(tf_object,feed_dict={a_ph: a_data[i].reshape([1,group_order]),b_ph: b_data[i].reshape([1,group_order])}).reshape((1,-1)),delimiter=',')

    def save_weights(tf_object,filename,remove_old=True):
	if remove_old and os.path.exists(filename):
	    os.remove(filename)
	with open(filename,'ab') as fout:
	    numpy.savetxt(fout,sess.run(tf_object),delimiter=',')

    def train_with_standard_loss():
	training_order = numpy.random.permutation(len(a_data))
	for example_i in training_order:
	    sess.run(train,feed_dict={eta_ph: curr_eta,a_ph: a_data[example_i].reshape([1,group_order]),b_ph: b_data[example_i].reshape([1,group_order]),target_ph: y_data[example_i].reshape([1,group_order])})

    def batch_train_with_standard_loss():
	sess.run(train,feed_dict={eta_ph: curr_eta,a_ph: a_data,b_ph: b_data,target_ph: y_data})

    print "Initial loss: %f" %(test_accuracy())

    #loaded_pre_outputs = numpy.loadtxt(pre_output_filename_to_load,delimiter=',')

    curr_eta = init_eta
    rep_track = []
    filename = filename_prefix + "rep_tracks.csv"
    #if os.path.exists(filename):
#	os.remove(filename)
    save_activations(a_rep,filename_prefix+"initial_a_reps.csv")
    save_activations(b_rep,filename_prefix+"initial_b_reps.csv")
#    fout = open(filename,'ab')
    for epoch in xrange(nepochs):
        batch_train_with_standard_loss()
	if epoch % 1000 == 0:
	    curr_loss = test_accuracy()
	    print "epoch: %i, loss: %f" %(epoch, curr_loss)	
	    if curr_loss < early_termination_loss:
		print "Stopping early!"
		break 
	if epoch % eta_decay_epoch == 0:
	    curr_eta *= eta_decay
	

    #print sess.run(output,feed_dict={eta_ph: curr_eta,a_ph: a_data,b_ph: b_data})
    max_dev = numpy.max(numpy.abs(y_data-sess.run(output,feed_dict={eta_ph: curr_eta,a_ph: a_data,b_ph: b_data})))
    final_loss = test_accuracy()
    print "Final loss: %f, maximum deviation: %f" %(final_loss,max_dev)

    if final_loss < 0.05 and max_dev < 0.2:
	save_activations(a_rep,filename_prefix+"final_a_reps.csv")
	save_activations(b_rep,filename_prefix+"final_b_reps.csv")
