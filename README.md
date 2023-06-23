# Automatic Differentiation with Tensor

## Introduction

Automatic differentiation (AD) is a technique used in mathematical optimization and machine learning to efficiently compute the derivatives of functions. It provides a way to automatically compute the gradients of complex functions by breaking them down into a sequence of elementary operations.

This project implements automatic differentiation in a custom type called `Tensor`. The `Tensor` type supports popular mathematical operations such as addition, multiplication, division, power, and negation, while also tracking the intermediate derivatives during computation. By leveraging this implementation, you can easily compute gradients and perform optimization tasks.

## Key Features

- Custom `Tensor` type with support for common mathematical operations.
- Efficient computation of gradients using reverse mode automatic differentiation.
- Demo notebook (`grad_from_scratch_demo.ipynb`) illustrating the basic idea and showcasing visualizations.
- `action.py` file that demonstrates minimizing a function and estimating a linear learnable function using the `Tensor` type, along with other types like `Module` and `Parameters`.
- `autograd` folder containing the implementation of the `Tensor` type and an optimizer.
- `autograd/module.py` file implementing a `Module` class for building learnable models.
- `autograd/parameter.py` file defining the `Parameters` class to hold model parameters.
- `tests` folder containing unit tests to validate the correctness of the implementation.

## How to Use

To get started with the project, follow these steps:

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/your-username/automatic-differentiation.git
   ```

2. Ensure you have the required dependencies installed. You can set up a virtual environment and install the dependencies by running:

   ```bash
   cd automatic-differentiation
   python -m venv env
   source env/bin/activate      # On Windows, use `env\Scripts\activate`
   pip install -r requirements.txt
   ```

3. Explore the demo notebook `grad_from_scratch_demo.ipynb` to understand the basic concepts of automatic differentiation and see visualizations of the process.

4. Examine the `action.py` file to understand how the `Tensor` type is used to minimize a function and estimate a linear learnable function.

5. Dig into the `autograd` folder to explore the implementation details:

   - `autograd/tensor.py` contains the implementation of the `Tensor` class with support for mathematical operations and reverse mode AD.
   - `autograd/optimizer.py` provides an optimizer class for updating model parameters using gradients.
   - `autograd/module.py` defines the `Module` class, which serves as a base class for building learnable models.
   - `autograd/parameter.py` defines the `Parameters` class to hold the learnable parameters of a model.

6. Run the unit tests located in the `tests` folder to verify the correctness of the implementation:

   ```bash
   pytest tests/
   ```

   All tests should pass without any failures.

## Additional Resources

If you're interested in diving deeper into the concepts behind automatic differentiation and how it is implemented, here are some recommended resources:

- [Wikipedia - Automatic Differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation)
- [Automatic Differentiation in Machine Learning: A Survey](https://arxiv.org/abs/1502.05767)

## License

This project is licensed under the [MIT License](https://github.com/your-username/automatic-differentiation/blob/main/LICENSE). Feel free to use and modify it according to your needs.

## Conclusion

Automatic differentiation is a powerful tool in mathematical optimization and machine learning, and this project provides a practical implementation of reverse mode automatic differentiation in the `Tensor` type. With this implementation, you can easily compute gradients and perform optimization tasks while having a clear understanding of the underlying mathematical concepts.
