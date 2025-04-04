# Financial Time Series Forecasting with Neural Networks

This project focuses on building and evaluating neural network models for forecasting financial time series. It involves comparing the predictive performance of neural networks against a traditional [AR-GARCH](https://en.wikipedia.org/wiki/Autoregressive_conditional_heteroskedasticity) model. The project was undertaken as part of the VBA for finance course at ESILV during the 2018-2019 academic year.

## Project Goals

* Develop a neural network to forecast financial time series.
* Implement an AR-GARCH model as a benchmark for comparison.
* Evaluate and compare the forecasting abilities of both models.

## Project Components

### 1. AR-GARCH Model

* **Description:** The AR-GARCH model is a combination of two components: an Autoregressive (AR) model for returns and a Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model for volatility.
* **Implementation:** The AR-GARCH model is implemented in Excel, using macros to automate the calculations.
* **Steps:**

    1.  Data input: Price data is obtained (e.g., from abcbourse.com) and stored in the first column.
    2.  Calculate price returns and store them in the second column.
    3.  Predicted returns are calculated using the AR part of the model.
    4.  Residuals of the AR part are calculated.
    5.  Volatility (σ) is calculated.
    6.  Innovation (ε) is calculated.
    7.  Log-likelihood is calculated.
    8.  The sum of log-likelihoods is calculated and minimized using Excel’s solver.
    9.  Model parameters are defined, including initial σ², GARCH parameters, and AR order (n).
    10. The empirical variance of the innovation is calculated and constrained to be close to 1.
* **Automation:** A macro is recorded to create a function that returns the optimal parameters for a given time interval.

### 2. Neural Network Models

* **Description:** Artificial Neural Networks (ANNs) are used to forecast financial time series.
* **Reference:** The project draws upon concepts from a research article on perceptrons.
* **Types of Neural Networks Implemented:**

    #### Simple Perceptron
    <img width="461" alt="Single_P" src="https://github.com/user-attachments/assets/db982c57-aee3-415b-b251-d429ca738f4e" />


    * A basic neural network model with inputs for price returns at times (t-1) and (t-2) ; X1 and X2 and an output for the forecasted price return Y at time t.
    * Uses a sigmoid function as the activation function: $$f(x) = \frac{1}{1 + \exp(-x)}$$ 

    * Weights are initialized randomly and updated using a gradient descent algorithm with a learning rate (e.g., 0.5).
    * Weight update rule: $w \leftarrow w - \sigma x e$, where $e = (y - s)y(1 - y)$
    * Implemented as a `SimplePerceptron` class in VBA with methods for output calculation (`output`) and weight updates (`updateWeights`).
    * The network is trained on a portion of the data, and forecasting is performed on the remaining part.
    * Input and output scaling is addressed to handle the difference between the sigmoid output range (0 to 1) and actual return values.

    #### Multi-Layer Perceptron
    <img width="376" alt="Multi" src="https://github.com/user-attachments/assets/ad5023c2-530e-46f7-af88-a22a2e9d4c24" />

    * A more complex network consisting of multiple interconnected perceptrons.
    * The architecture includes input layers, hidden layers, and an output layer.
    * For simplicity, the project uses an architecture with \(n\) inputs, \(p\) hidden layers each with \(n\) nodes, and a single-node output layer.
    * Weights between layers are stored in a 3D array w<sub>i,j,k</sub>
    * Backpropagation is used to update weights, starting from the output layer and working backward.
    * Sensitivity of the error is calculated for each node in each layer.
    * A new class is created for the multi-layer perceptron.
    * Training and forecasting are performed similarly to the simple perceptron.

    #### Learning Rule Enhancements

    * **Momentum:** A technique to smooth weight updates by incorporating the previous weight change.

        \* Δw<sub>i,j,k</sub>(t) = -λ \* η \* z<sub>i,k-1</sub>(t) \* e<sub>i,k</sub>(t) + (1 - λ) \* Δw<sub>i,j,k</sub>(t-1)

    * **Non-constant learning rates:** Implementation of adaptive learning rates.

### 3. Comparison and Evaluation

* **Accuracy Functions:** The models’ performance is evaluated using accuracy functions such as \(R^2\), MPE (Mean Percentage Error), and MSRE (Mean Squared Relative Error).
* **Profit and Loss (P&L) Strategy:** A trading strategy is simulated where buying occurs when the forecasted return is positive, and selling occurs otherwise.
* **Result Analysis:** A dedicated worksheet is created for each model (AR-GARCH, simple perceptron, multi-layer perceptron) to analyze and compare the results.
* **User-Defined Metaparameters:** A user form is implemented to allow users to specify metaparameters such as:

    * Learning and testing sample dates.
    * Number of lags in the AR-GARCH model and ANN.
    * Parameters used for the ANN learning rule.

## Submission

* The project deliverables include the Excel file containing the implementations and any necessary supporting information.
* The submission deadline was **Thursday, December 27th**.

## Project Elements

* [Subject Outline](https://github.com/Hugo-Perr/2018-2019-ESILV_Neural_Network_VBA/blob/master/Subject_Outlines.pdf): Instructions & Requirements from the course.
* [Neural Network VBA](https://github.com/Hugo-Perr/2018-2019-ESILV_Neural_Network_VBA/blob/master/Neural_Network_VBA.xlsm): Final project submitted as an Excel document.

## Results

Data & AR-GARCH Model Parameters
<img width="1410" alt="Data_Arcelor" src="https://github.com/user-attachments/assets/5ef56694-879b-4972-b05e-6dacf1c593be" />

Simple Perceptron
<img width="1425" alt="Result_Simple" src="https://github.com/user-attachments/assets/443737d9-8c1a-4a87-aafb-5bedbb3d0eb5" />

Multi-Layer Perceptron
<img width="1306" alt="Result_Multi" src="https://github.com/user-attachments/assets/ebf4a1ac-cb55-4ee3-a5bf-e87cf6c7d1ca" />

