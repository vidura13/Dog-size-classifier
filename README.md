
# Dog Size Classifier

This project uses machine learning to predict a dog breed's **size category** (`small`, `medium`, `large`) based on personality and physical traits like barking, lifespan, shedding, energy, trainability, protectiveness, and friendliness.

---

## 📁 Dataset

- Source: [Dog Breeds by Personality and Size (Kaggle)](https://www.kaggle.com/datasets/frtgnn/dog-breeds-by-personality-and-size)
- The dataset includes 20 dog breeds with traits and breed group information.

---

## 🧠 Model

- Algorithm: Random Forest Classifier
- Input Features: `energy_level`, `friendliness`, `trainability`, `shedding`, `barking`, `lifespan`, and one-hot encoded `breed_group`s.
- Target: `size` category encoded as small, medium, or large.

---

## 💻 How to Use

1. Clone the repository.
2. Install required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Launch the Streamlit web app:

    ```bash
    streamlit run app.py
    ```

4. Use the sliders in the app to input dog traits and get the predicted dog size.

---

## 🗂️ Project Structure
- dog-breed-classifier/
- data/
dog_breeds_traits.csv
- models/
rf_model.pkl,
size_encoder.pkl,
feature_columns.pkl 

- notebooks/
dog_breed_size_predictor.ipynb

- app.py 
- README.md

---

## 🛠️ Requirements

- pandas
- numpy
- scikit-learn
- matplotlib
- streamlit
- joblib

---

## 📌 Notes

- The dataset is small, so model accuracy may vary.
- The app currently predicts only the dog size, not the breed.

---


