#include <iostream>
#include <vector>
#include <math.h>

// Define data types to save time and make code 
// look nicer and more readable
typedef std::vector<std::vector<int>> in_set_t;
typedef std::vector<int> labels_t;
typedef std::vector<double> weights_t;
typedef std::vector<double> dot_t;


double rnd_double(double min, double max) {
    // calculates a random decimal number given a min and a max number
    double rnd = (double)rand() / RAND_MAX;
    return min + rnd * (max - min);
}

dot_t dot_product(in_set_t sets, weights_t weights, double bias = 0) {
    // Caclulates the dot product with and without a bias
    dot_t products;
    double product = 0;
    products.empty();
    for (int i = 0; i < sets.size(); i++) {
        for (int j = 0; j < sets[i].size(); j++){
            product = product + sets[i][j] * weights[j] + bias;
        }
        products.push_back(product);
        product = 0;
    }
    
    return products;
}

dot_t calculate_error(dot_t z, labels_t labels) {
    // Caclulates the deviation from the label's value
    for (int i = 0; i < z.size(); i++ ) {
        z[i] -= labels[i];
    }
    return z;
}

in_set_t invert_inputs (in_set_t inputs) {
    // Inverts a 2D array of inputs
    in_set_t inverted_inputs;
    inverted_inputs.empty();
    inverted_inputs.resize(inputs[0].size());
    for (int i = 0; i < inputs.size (); i++) {
        for (int j = 0; j < inputs[i].size (); j++) {
            inverted_inputs[j].push_back(inputs[i][j]);
        }
    }
    return inverted_inputs;
}

dot_t sigmoid(dot_t x) {
    // Sigmoid Function that returns a dot product 
    dot_t sig;
    sig.empty();
    for (int i = 0; i < x.size(); i++) {
        sig.push_back(1/(1 + exp( -(x[i]) )));
    }
    return sig;
}

double sigmoid(int x) {
    // Sigmoid Function that returns a double
    return 1/(1 + exp( -x ));
}

dot_t sigmoid_derivatve(dot_t x) {
    // The derivative of the Sigmoid Function that returns a dot product
    dot_t sig_der;
    sig_der.empty();
    for (int i = 0; i < x.size(); i++) {
        double sig = sigmoid(x[i]);
        sig_der.push_back( sig * (1 - sig));
    }
    return sig_der;
}

void update_weights(weights_t &weights, double learning_rate, in_set_t inputs, dot_t z_del) {
    // Updates weights based on the input set and the z_del (see train function for explaniation)
    dot_t temp = dot_product(inputs, z_del);

    for(int i = 0; i < weights.size(); i++ ) {
        weights[i] -= learning_rate * temp[i];
    }
}

void fill_rnd(std::vector<double> &vector_array, int size) {
    // Fill a vector with random numbers
    vector_array.empty();
    for (int i = 0; i < size; i++) {
        vector_array.push_back(rnd_double(0,2));
    }
}

double sum_error(dot_t error) {
    // get's total error by summing the dot product
    double sum = 0;
    for (int  i = 0; i < error.size(); i++ ) {
        sum += error[i];
    }
    return sum;
}

void train(in_set_t input_set, labels_t labels, weights_t &weights, double &bias, double learning_rate) {
    // define number of times to train
    int max_training_rounds = 25000;
    for(int i = 0; i < max_training_rounds; i++) {
        // copy input_set to inputs
        in_set_t inputs = input_set;
        
        // get the dot product of inputs and weights with a bias
        // Also the first step in Feedforward
        dot_t input_weight_dot = dot_product(inputs, weights, bias);

        // second part of Feedforward 
        // passing the dot product to the sigmoid function
        dot_t z = sigmoid(input_weight_dot);

        // Caclulate the error by subtracting the second
        // phase of the Feedforward by the labels
        dot_t error = calculate_error(z, labels);

        double summed_error = sum_error(error);
        
        // Print summed error very 100 runs
        if (i%100 == 0) {
            std::cout << "Total error for training run #" << i << " is: " << summed_error << std::endl;
        }

        // Caclualte The Slope
        // slope = input * d_cost * d_pred
        dot_t d_cost = error;
        dot_t d_pred = sigmoid_derivatve(z);
        
        // z_del is the dot product of d_cost and d_pred
        dot_t z_del;
        z_del.empty();
        for (int i = 0; i < d_cost.size(); i++ ) {
            z_del.push_back(d_cost[i] * d_pred[i]);
        }

        // update the weights
        inputs = invert_inputs(inputs);
        update_weights(weights, learning_rate, inputs, z_del);
        
        //update the bias
        for (int i = 0; i < z_del.size(); i++ ) {
            bias = bias - learning_rate * z_del[i];
        }
    }
}

void print_weights (weights_t weights) {
    // Prints weights
    std::cout << "[";
    for (int i = 0; i < weights.size (); i++) {
        if (i < weights.size () - 1) {
            std::cout << weights[i] << ", ";
	    }
        else {
	      std::cout << weights[i] << "]" << std::endl;
	    }
	}
}

std::string set_to_string(in_set_t input_set) {
    // turns a input set into a string
    std::string temp = "[";
    for (int i = 0; i < input_set[0].size(); i++) {
        if (i < input_set[0].size() - 1) {
            temp = temp + std::to_string(input_set[0][i]) + ", ";
        }
        else {
            temp = temp + std::to_string(input_set[0][i]) + "]";
        }
    }
    return temp;
}

void test_ann(in_set_t input_set, weights_t weights, double bias) {
    // function to help test the ANN after and before training
    dot_t result = sigmoid(dot_product(input_set, weights, bias));
    double confidence;
    if(result[0] > 0.85) {
        confidence = result[0];
    } 
    else {
        confidence = 1 - result[0];
    }
    std::cout << "the result for " << set_to_string(input_set) << " is: ";
    std::cout << (result[0] > 0.85 ? "1 (Bright) ":"0 (Dark) ") << "\t with a confidence of " << confidence << std::endl;
}

int main() {

    // init rand
    srand (time(NULL));

    // Training set
    in_set_t input_set = {
        {0, 0, 0, 0},  //  0 Dark
        {0, 0, 0, 1},  //  0 Dark
        {0, 0, 1, 0},  //  0 Dark
        {0, 0, 1, 1},  //  1 Bright
        {0, 1, 0, 0},  //  0 Dark
        {0, 1, 0, 1},  //  1 Bright
        {0, 1, 1, 0},  //  1 Bright
        {0, 1, 1, 1},  //  1 Bright
        {1, 0, 0, 0},  //  0 Dark
        {1, 0, 0, 1},  //  1 Bright
        {1, 0, 1, 0},  //  1 Bright
        {1, 0, 1, 1},  //  1 Bright
        {1, 1, 0, 0},  //  1 Bright
        {1, 1, 0, 1},  //  1 Bright
        {1, 1, 1, 0},  //  1 Bright
        {1, 1, 1, 1}}; //  1 Bright

    // test set
    in_set_t testing_set = {
        {0, 0, 0, 0},  //  0 Dark
        {0, 1, 1, 0},  //  1 Bright
        {0, 0, 1, 0},  //  0 Dark
        {1, 0, 1, 1},  //  1 Bright
        {1, 1, 0, 0},  //  1 Bright
        {1, 1, 0, 1},  //  1 Bright
        {1, 1, 1, 0},  //  1 Bright
        {1, 1, 1, 1}   //  1 Bright
    };

    // Labels for training set
    labels_t labels = {0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1};

    // init weights to random values
    weights_t weights;
    fill_rnd(weights, 4);

    // init bias with a random value
    double bias = rnd_double(0,2);

    // define leaening rate to 50%
    double learning_rate = 0.5;


    // show ANN results before training
    std::cout << std::endl << "Weights Before Training: ";
    print_weights(weights);
    std::cout << "Bias Before Training:" << bias << std::endl << std::endl;

    std::cout << "Testing ANN Before Training: " << std::endl << std::endl;
    for(int i = 0; i < input_set.size(); i++) {
        in_set_t test = {input_set[i]};
        test_ann(test, weights, bias);
    }

    // train ANN
    train(input_set, labels, weights, bias, learning_rate);
    

    // show ANN results after training
    std::cout << std::endl << "Weights After Training: ";
    print_weights(weights);
    std::cout << "Bias After Training:" << bias << std::endl << std::endl;

    std::cout << "Testing ANN After Training: " << std::endl << std::endl;
    for(int i = 0; i < input_set.size(); i++) {
        in_set_t test = {input_set[i]};
        test_ann(test, weights, bias);
    }

    return 0;
}