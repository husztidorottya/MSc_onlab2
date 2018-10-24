import tensorflow as tf 
import numpy as np
import helpers
import operator
import random
import argparse
import os
from sklearn.model_selection import train_test_split

# read all the pictures from a directory
def read_directory(path, signatures, timeStamp, target):
    for file in os.listdir(path):
        # read a signature's descriptors
        firstLine = 1
        signature = [2] # SOS
        coordinates = []

        with open(path + '/' + file) as ifile:
            for line in ifile:
                if firstLine == 0:
                    splitedLine = line.split(' ')
                    x = splitedLine[0]
                    y = splitedLine[1]
                    # create tuple from coordinates and add to the array
                    signature.append(int(x))
                    signature.append(int(y))
                    coordinates.append((x,y))
                    timeStamp.append(splitedLine[2])
                else:
                    firstLine = 0
                    
            signature.append(1) # EOS
            signatures.append([signature, len(signatures)])
                     
        # vizsgálandó pontok (pixelek) tömbje, elemei az azonos (x,y) koordinátájú pixelek
        samePixels = []
        
        # kigyűjteni az azonos (x,y) párok indexeit
        for i in range(0,len(coordinates)):
            if coordinates[i] != (-1,-1):
                samePixel = []
                for j in range(i+1,len(coordinates)):
                    if coordinates[i] == coordinates[j]:
                        # if it doesn't have any elements
                        if len(samePixel) == 0:
                            # add 3 to index to start from 3, because 0 = pad, 1 = eos, 2 = sos
                            samePixel.append(i+3)
                            samePixel.append(j+3)
                        else:
                            samePixel.append(j)
                
                        coordinates[j] = (-1,-1)
                    
                if len(samePixel) != 0:
                    samePixels.append(samePixel)
        
        # create a single list from array of lists
        samePixels2 = [2] # SOS
        for r in samePixels:
            for s in r:
                samePixels2.append(s)
                
        target.append(samePixels2)
##########

# create batches with size of batch_size
def create_batches(source_data, target_data, parameters, target):
    # stores batches
    source_batches = []
    target_batches = []
    # stores last batch ending index
    prev_batch_end = 0
    
    for j in range(0, len(source_data)):
	# if it's a full batch
        if j % parameters.batch_size == 0 and j != 0:
            # stores a batch
            sbatch = []
            tbatch = []
            for k in range(prev_batch_end+1,j+1):
                # store sequence
                sbatch.append(source_data[k][0])
                # store expected target_data (known from source_data index)
                #tbatch.append(target_data[source_data[k][1]])
                tbatch.append(target[source_data[k][1]])
            # add created batch
            source_batches.append(sbatch)
            target_batches.append(tbatch)
            prev_batch_end = j
            
    # put the rest of it in another batch
    if prev_batch_end != j:
        sbatch = []
        tbatch = []
        for k in range(prev_batch_end+1,j):
            sbatch.append(source_data[k][0])
            # tbatch.append(target_data[source_data[k][1]])
            tbatch.append(target[source_data[k][1]])
        source_batches.append(sbatch)
        target_batches.append(tbatch)

    # in case its a single line
    if j == 0: 
        source_batches.append([source_data[j][0]])
        # target_batches.append([target_data[source_data[j][1]])
        target_batches.append([target[source_data[j][1]]])

    return source_batches, target_batches



# feed encoder with the sequences of the next batch
def next_feed(source_batches, target_batches, encoder_inputs, encoder_inputs_length, decoder_targets, batch_num, parameters, learning_rate):
        # get transpose of source_batches[batch_num]
        #print('next_feed:',helpers.batch(source_batches[batch_num]))
        encoder_inputs_, encoder_input_lengths_ = helpers.batch(source_batches[batch_num])
    
    
        # get max input sequence length
        max_input_length = max(encoder_input_lengths_)
    
        # target word is max character_changing_num character longer than source word 
        # get transpose of target_batches[i] and put an EOF and PAD at the end
        decoder_targets_, _ = helpers.batch(
            [(sequence) + [parameters.EOS] + [parameters.PAD] * ((max_input_length + parameters.character_changing_num - 1) - len(sequence))  for sequence in target_batches[batch_num]]
        )
        
        
        
        return {
            encoder_inputs: encoder_inputs_,
            encoder_inputs_length: encoder_input_lengths_,
            decoder_targets: decoder_targets_ #,
            #learning_rate: parameters.learning_rate
        }
    
# feeds the encoder with the next batch size sequences
def next_feed2(source_batches, target_batches, encoder_inputs, encoder_inputs_length, decoder_targets, parameters):
    # get transpose of source_batches[batch_num]
    
    #print('next_feed2:',helpers.batch(source_batches))
    encoder_inputs_, encoder_input_lengths_ = helpers.batch(source_batches)
    
    
    # get max input sequence length
    max_input_length = max(encoder_input_lengths_)
    
    # target word is max character_changing_num character longer than source word 
    # get transpose of target_batches[i] and put an EOF and PAD at the end
    decoder_targets_, _ = helpers.batch(
            [(sequence) + [parameters.EOS] + [parameters.PAD] * ((max_input_length + parameters.character_changing_num - 1) - len(sequence))  for sequence in target_batches]
    )
   
    return {
            encoder_inputs: encoder_inputs_,
            encoder_inputs_length: encoder_input_lengths_,
            decoder_targets: decoder_targets_,
        }, max_input_length


# train the model with chosen parameters
def train_model(source_data, target_data, encoder_inputs, encoder_inputs_length, decoder_targets, train_op, loss, decoder_prediction, sess, loss_track, parameters, saver, learning_rate, summ, target):
    # early stopping patience
    patience_counter = 0

    for epoch_num in range(0,150):
    #for epoch_num in range(parameters.epoch):
            print('Epoch:',epoch_num)
            epoch_loss = 0
            
            # shuffle it in every epoch for creating random batches
            source_data = random.sample(source_data, len(source_data))
        
            # encoder inputs and decoder outputs devided into batches
            source_batches, target_batches = create_batches(source_data, target_data, parameters, target)
            
            # get every batches and train the model on it
            for batch_num in range(0, len(source_batches)):
                fd = next_feed(source_batches, target_batches, encoder_inputs, encoder_inputs_length, decoder_targets, batch_num, parameters, learning_rate)
   
                
                _, l = sess.run([train_op, loss], fd)
                epoch_loss += l

                '''
                predict_ = sess.run(decoder_prediction, fd)
                for i, (dec, pred) in enumerate(zip(fd[decoder_targets].T, predict_.T)):
                    print('predicted > {}'.format(pred))
                    print('target output: {}'.format(dec))
                    if i >= 2:
                        break
                print()
                '''
                
            print('epoch:',epoch_num, 'loss:', epoch_loss)

            # store current epoch loss to calculate early stopping delta
            loss_track.append(epoch_loss)

            '''
            # early stopping
            if epoch_num > 0 and loss_track[epoch_num - 1] - loss_track[epoch_num] > parameters.early_stopping_delta:
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter > parameters.early_stopping_patience:
                break
            '''

    return


# class stores model parameters
class Parameters:
    def __init__(self, SOS, EOS, PAD, character_changing_num, input_embedding_size, neuron_num, epoch, delta, patience, batch_size, learning_rate):
        self.SOS = SOS
        self.EOS = EOS
        self.PAD = PAD
        self.character_changing_num = character_changing_num
        self.input_embedding_size = input_embedding_size
        self.neuron_num = neuron_num
        self.epoch = epoch
        self.early_stopping_delta = delta
        self.early_stopping_patience = patience
        self.batch_size = batch_size
        self.learning_rate = learning_rate


def largest(arr,n): 
  
    #print(arr[0])
    # Initialize maximum element  
    max = arr[0]
  
    # Traverse array elements from second 
    # and compare every element with  
    # current max 
    for i in range(1, n): 
        if arr[i] > max: 
            max = arr[i] 
    return max


########
# read all the pictures from a directory
def read_directory2(path, signatures, timeStamp, target):
    
    for file in os.listdir(path):
        # read a signature's descriptors
        firstLine = 1
        signature = [2] # SOS
        coordinates = []

        with open(path + '/' + file) as ifile:
            for line in ifile:
                if firstLine == 0:
                    splitedLine = line.split(' ')
                    x = splitedLine[0]
                    y = splitedLine[1]
                    # create tuple from coordinates and add to the array
                    signature.append(int(x))
                    signature.append(int(y))
                    coordinates.append((x,y))
                    timeStamp.append(splitedLine[2])
                else:
                    firstLine = 0
                    
            signature.append(1) # EOS
            signatures.append(signature)
                     
        # vizsgálandó pontok (pixelek) tömbje, elemei az azonos (x,y) koordinátájú pixelek
        samePixels = []
        
        # kigyűjteni az azonos (x,y) párok indexeit
        for i in range(0,len(coordinates)):
            if coordinates[i] != (-1,-1):
                samePixel = []
                for j in range(i+1,len(coordinates)):
                    if coordinates[i] == coordinates[j]:
                        # if it doesn't have any elements
                        if len(samePixel) == 0:
                            # add 3 to index to start from 3, because 0 = pad, 1 = eos, 2 = sos
                            samePixel.append(i+3)
                            samePixel.append(j+3)
                        else:
                            samePixel.append(j)
                
                        coordinates[j] = (-1,-1)
                    
                if len(samePixel) != 0:
                    samePixels.append(samePixel)
        
        # create a single list from array of lists
        samePixels2 = [2] # SOS
        for r in samePixels:
            for s in r:
                samePixels2.append(s)
                
        target.append(samePixels2)
        
##########


##########


def main():
    #################
    timeStamp = []
    firstLine = 1

    # CONSTANTS
    # stores all versions of a signature
    signatures = []   
    # stores loop interval
    target = []
    #################

    
    parameters = Parameters(2, 1, 0, 10, 300, 100, 100, 0.001, 5, 16, 0.001)

    batch_size = parameters.batch_size

    loss_track = []
    
    #source_data, target_data = read_split_encode_data('./task1.tsv', alphabet_and_morph_tags, parameters)
    read_directory('./Task2/U1', signatures, timeStamp, target)
    
    # Clears the default graph stack and resets the global default graph.
    tf.reset_default_graph() 

    with tf.Session() as sess:
        legnagyobbak = []
        for i in range(0,len(signatures)):
            legnagyobbak.append(largest(signatures[i][0],len(signatures[i][0])))
    
        vocab_size_src = largest(legnagyobbak,len(legnagyobbak))
        # calculate vocab_size
        vocab_size = vocab_size_src + 1
        
        encoder_hidden_units = parameters.neuron_num 
        decoder_hidden_units = encoder_hidden_units * 2 
        encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
        encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
        decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')


        embeddings = tf.Variable(tf.eye(vocab_size, parameters.input_embedding_size), dtype='float32', name='embeddings')

        encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)

        # define encoder
        encoder_cell = tf.contrib.rnn.GRUCell(encoder_hidden_units)

        # define bidirectionel function of encoder (backpropagation)
        ((encoder_fw_outputs,
        encoder_bw_outputs),
        (encoder_fw_final_state,
        encoder_bw_final_state)) = (
        tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                    cell_bw=encoder_cell,
                                    inputs=encoder_inputs_embedded,
                                    sequence_length=encoder_inputs_length,
                                    dtype=tf.float32, time_major=True)
        )

        # Concatenates tensors along one dimension.
        encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

        # because by GRUCells the state is a Tensor, not a Tuple like by LSTMCells
        encoder_final_state = tf.concat(
            (encoder_fw_final_state, encoder_bw_final_state), 1)

        # define decoder
        decoder_cell = tf.contrib.rnn.GRUCell(decoder_hidden_units)

        #we could print this, won't need
        encoder_max_time, parameters.batch_size = tf.unstack(tf.shape(encoder_inputs))
        # (character_changing_num-1) additional steps, +1 leading <EOS> token for decoder inputs
        decoder_lengths = encoder_inputs_length + parameters.character_changing_num

        #manually specifying since we are going to implement attention details for the decoder in a sec
        #weights
        W = tf.Variable(tf.eye(decoder_hidden_units, vocab_size), dtype='float32', name='W')
        #bias
        b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32, name='b')

        #create padded inputs for the decoder from the word embeddings
        #were telling the program to test a condition, and trigger an error if the condition is false.
        assert parameters.EOS == 1 and parameters.PAD == 0 and parameters.SOS == 2

        sos_time_slice = tf.fill([parameters.batch_size], 2, name='SOS')
        eos_time_slice = tf.ones([parameters.batch_size], dtype=tf.int32, name='EOS')
        pad_time_slice = tf.zeros([parameters.batch_size], dtype=tf.int32, name='PAD')

        # send batch size sequences into encoder at one time
        parameters.batch_size = batch_size

        #retrieves rows of the params tensor. The behavior is similar to using indexing with arrays in numpy
        sos_step_embedded = tf.nn.embedding_lookup(embeddings, sos_time_slice)
        eos_step_embedded = tf.nn.embedding_lookup(embeddings, eos_time_slice)
        pad_step_embedded = tf.nn.embedding_lookup(embeddings, pad_time_slice)
    
        #manually specifying loop function through time - to get initial cell state and input to RNN
        #normally we'd just use dynamic_rnn, but lets get detailed here with raw_rnn

        #we define and return these values, no operations occur here
        def loop_fn_initial():
            initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
            initial_input = sos_step_embedded
            #last time steps cell state
            initial_cell_state = encoder_final_state
            #none
            initial_cell_output = None
            # none
            initial_loop_state = None  # we don't need to pass any additional information
            return (initial_elements_finished,
                initial_input,
                initial_cell_state,
                initial_cell_output,
                initial_loop_state)

        #attention mechanism --choose which previously generated token to pass as input in the next timestep
        def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):

            def get_next_input():
                #dot product between previous ouput and weights, then + biases
                output_logits = tf.add(tf.matmul(previous_output, W), b)
                #Logits simply means that the function operates on the unscaled output of 
                #earlier layers and that the relative scale to understand the units is linear. 
                #It means, in particular, the sum of the inputs may not equal 1, that the values are not probabilities 
                #(you might have an input of 5).
                #prediction value at current time step
        
                #Returns the index with the largest value across axes of a tensor.
                prediction = tf.argmax(output_logits, axis=1)
                #embed prediction for the next input
                next_input = tf.nn.embedding_lookup(embeddings, prediction)
            
                return next_input
    
            elements_finished = (time >= decoder_lengths) # this operation produces boolean tensor of [batch_size]
                                                  # defining if corresponding sequence has ended
    
            #Computes the "logical and" of elements across dimensions of a tensor.
            finished = tf.reduce_all(elements_finished) # -> boolean scalar
            #Return either fn1() or fn2() based on the boolean predicate pred.
            input = tf.cond(finished, lambda: pad_step_embedded, get_next_input)
    
            #set previous to current
            state = previous_state
            output = previous_output
            loop_state = None

            return (elements_finished, 
                input,
                state,
                output,
                loop_state)

        def loop_fn(time, previous_output, previous_state, previous_loop_state):
            # time == 0
            if previous_state is None:
                assert previous_output is None and previous_state is None
                return loop_fn_initial()
            else:
                return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

        #Creates an RNN specified by RNNCell cell and loop function loop_fn.
        #This function is a more primitive version of dynamic_rnn that provides more direct access to the 
        #inputs each iteration. It also provides more control over when to start and finish reading the sequence, 
        #and what to emit for the output.
        #ta = tensor array
        decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
        decoder_outputs = decoder_outputs_ta.stack()

        #Unpacks the given dimension of a rank-R tensor into rank-(R-1) tensors.
        #reduces dimensionality
        decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
        #flettened output tensor
        decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
        #pass flattened tensor through decoder
        decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
        #prediction vals
        decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, vocab_size))

        #final prediction
        decoder_prediction = tf.argmax(decoder_logits, 2)

        #cross entropy loss
        #one hot encode the target values so we don't rank just differentiate
        stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
            logits=decoder_logits,
        )

        #loss function
        loss = tf.reduce_mean(stepwise_cross_entropy)
            
        
        learning_rate = parameters.learning_rate

        #train it 
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        sess.run(tf.global_variables_initializer())

        try:
            saver = tf.train.Saver()

            summ = tf.summary.merge_all()
            writer = tf.summary.FileWriter('output', sess.graph)

            X_train, X_test, Y_train, Y_test = train_test_split(signatures,target,test_size=0.2, random_state=42)
            
            train_model(X_train, Y_train, encoder_inputs, encoder_inputs_length, decoder_targets, train_op, loss, decoder_prediction, sess, loss_track, parameters,saver, learning_rate, summ, target)
            
            ####################
            #### TEST ACCURACY
            ####################
            #timeStamp = []
            #firstLine = 1
            # CONSTANTS
            # stores all versions of a signature
            #signatures = []   
            # stores loop interval
            #target = []
            
            #read_directory2('./Task2/U1', signatures, timeStamp, target)

            # hogy elhagyja azt, hogy melyik y tartozik hozza, csak a szekvencia legyen benne
            x_test = []
            for v in X_test:
                x_test.append(v[0])

            fd, max_input_length = next_feed2(x_test, Y_test, encoder_inputs, encoder_inputs_length, decoder_targets, parameters)
            
            # get decoder predictions
            predict_ = sess.run(decoder_prediction, fd)

            '''            
            for i, (dec, pred) in enumerate(zip(fd[decoder_targets].T, predict_.T)):
                print('predicted > {}'.format(pred))
                print('target output: {}'.format(dec))
            
            print()
            '''

            decoder_targets_, _ = helpers.batch(
            		[(sequence) + [parameters.EOS] + [parameters.PAD] * ((max_input_length + parameters.character_changing_num - 1) - len(sequence))  for sequence in Y_test]
        	)


            # printing prediction on test data set           
            targets = decoder_targets_.transpose()
            predictions = predict_.transpose()

            for t in range(0,len(targets)):
                print('target:',targets[t])
                print('predicted:',predictions[t])

            correct = tf.equal(tf.cast(predict_.transpose(), tf.float32), tf.cast(decoder_targets_.transpose(), tf.float32))
            equality = correct.eval(fd)

            samplenum = 0
            sampleright = 0
		
            # analises predicted words for the percentage of full word equality
            for i in equality:
                right = 1
                for j in i:
                    if j == False:
                        right = 0
                        break
                if right == 1:
                    sampleright += 1
                samplenum += 1

            print('accuracy:',sampleright/samplenum)
            
        except KeyboardInterrupt:
            print('training interrupted')


if __name__ == '__main__':
    main()

