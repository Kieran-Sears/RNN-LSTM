import scala.math._
import scala.util.Random
import scala.collection.mutable.ArraySeq

object LSTM {
  Random.setSeed(0)

  // hyper-parameters
  val iterations = 10000;
  val binary_dim = 8
  val max = 20
  //val largest_number = pow(2, binary_dim)

  // input variables
  val alpha = 0.1
  val input_dim = 2
  val hidden_dim = 16
  val output_dim = 1


  def main(args: Array[String]): Unit = {

    // specify the dimensions of the layers
    val synapse_0 = Array.tabulate(input_dim, hidden_dim) (f)
    val synapse_1 = Array.tabulate(input_dim, hidden_dim) (f)
    val synapse_h = Array.tabulate(input_dim, hidden_dim) (f)

    train(synapse_0, synapse_1, synapse_h, 0)
  }


  def f(i: Int, j: Int): Double = 2 * Random.nextDouble - 1


  // logistic function (check wiki sigmoid function for more):
  // simple implementation for demonstrations sake
  def sigmoid(x: Double): Double = 1 / (1 + exp(-x))

  def sigmoid_output_to_derivative(output: Double): Double =
    output * (1 - output)

  // training dataset generation
  def intToBinary(int: Int): ArraySeq[Boolean] = {
    def accBits(num: Int, acc: ArraySeq[Boolean], i: Int): ArraySeq[Boolean] = {
      if (num < 1) acc
      if (num % 2 == 0) {
        acc(i) = false
        accBits(num / 2, acc, i + 1)
      } else {
        acc(i) = true
        accBits(num / 2, acc, i + 1)
      }
    }
      accBits(int, ArraySeq[Boolean](), 0)
  }



  val synapse_0_update = Array.tabulate(input_dim, hidden_dim) ((_,_) => 0)
  val synapse_1_update = Array.tabulate(input_dim, hidden_dim) ((_,_) => 0)
  val synapse_h_update = Array.tabulate(input_dim, hidden_dim) ((_,_) => 0)



  def generateAdditionValues(): (Int, Int, Int) = {
    val a_int = Random.nextInt
    val b_int = Random.nextInt
    (a_int, b_int,  a_int + b_int)
  }

  def train(synapse0: Array[Array[Double]], synapse1: Array[Array[Double]], synapseh: Array[Array[Double]], overallError: Int): Boolean = {

    val add =  generateAdditionValues()

    val X = Array.ofDim[Boolean](3,3,3)
       (ArraySeq(ArraySeq(intToBinary(add._1))))
  // val X = Array.ofDim(3)(ArraySeq(ArraySeq(intToBinary(add._1))))

    // X = np.array([[a[binary_dim - position - 1],b[binary_dim - position - 1]]])
    // y = np.array([[c[binary_dim - position - 1]]]).T

    /*
     X is the same as "layer_0" in the pictures.
     X is a list of 2 numbers, one from a and one from b.
     It's indexed according to the "position" variable,
     but we index it in such a way that it goes from
     right to left. So, when position == 0, this is the
     farhest bit to the right in "a" and the farthest bit
     to the right in "b". When position equals 1, this
     shifts to the left one bit.
    */
    true
  }



}


//# training logic
//for j in range(10000):
//
//# generate a simple addition problem (a + b = c)
//a_int = np.random.randint(largest_number/2) # int version
//a = int2binary[a_int] # binary encoding
//
//b_int = np.random.randint(largest_number/2) # int version
//b = int2binary[b_int] # binary encoding
//
//# true answer
//c_int = a_int + b_int
//c = int2binary[c_int]
//
//# where we'll store our best guess (binary encoded)
//d = np.zeros_like(c)
//
//overallError = 0
//
//layer_2_deltas = list()
//layer_1_values = list()
//layer_1_values.append(np.zeros(hidden_dim))
//
//# moving along the positions in the binary encoding
//for position in range(binary_dim):
//
//# generate input and output
//
//
//# hidden layer (input ~+ prev_hidden)
//layer_1 = sigmoid(np.dot(X,synapse_0) + np.dot(layer_1_values[-1],synapse_h))
//
//# output layer (new binary representation)
//layer_2 = sigmoid(np.dot(layer_1,synapse_1))
//
//# did we miss?... if so, by how much?
//layer_2_error = y - layer_2
//layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(layer_2))
//overallError += np.abs(layer_2_error[0])
//
//# decode estimate so we can print it out
//d[binary_dim - position - 1] = np.round(layer_2[0][0])
//
//# store hidden layer so we can use it in the next timestep
//layer_1_values.append(copy.deepcopy(layer_1))
//
//future_layer_1_delta = np.zeros(hidden_dim)
//
//for position in range(binary_dim):
//
//X = np.array([[a[position],b[position]]])
//layer_1 = layer_1_values[-position-1]
//prev_layer_1 = layer_1_values[-position-2]
//
//# error at output layer
//layer_2_delta = layer_2_deltas[-position-1]
//# error at hidden layer
//layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)
//
//# let's update all our weights so we can try again
//synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
//synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
//synapse_0_update += X.T.dot(layer_1_delta)
//
//future_layer_1_delta = layer_1_delta
//
//
//synapse_0 += synapse_0_update * alpha
//synapse_1 += synapse_1_update * alpha
//synapse_h += synapse_h_update * alpha
//
//synapse_0_update *= 0
//synapse_1_update *= 0
//synapse_h_update *= 0
//
//# print out progress
//if(j % 1000 == 0):
//print "Error:" + str(overallError)
//print "Pred:" + str(d)
//print "True:" + str(c)
//out = 0
//for index,x in enumerate(reversed(d)):
//out += x*pow(2,index)
//print str(a_int) + " + " + str(b_int) + " = " + str(out)
//print "------------"
