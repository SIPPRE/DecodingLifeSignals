<a id="readme-top"></a>

<!-- omit in toc -->
# Decoding Life Signals: Innovations in Biomedical Engineering

Spring Semester 2025 - EUNICE Course

<!-- omit in toc -->
## Table of Contents
- [Course Description](#course-description)
  - [Environment Setup](#environment-setup)
  - [Key Learning Objectives](#key-learning-objectives)
  - [Course Highlights](#course-highlights)
  - [Prerequisites](#prerequisites)
- [Tools and Technologies](#tools-and-technologies)
- [Weekly Materials](#weekly-materials)
- [Resources](#resources)
- [Contact](#contact)

## Course Description
This course explores the fascinating world of **biomedical signal processing**, focusing on understanding and analyzing the electrical signals that our bodies generate. From brain waves to heart rhythms, students will learn to **decode**, **analyze**, and **interpret** various biological signals using modern computational techniques.

The course combines **theoretical foundations with hands-on practical experience**, utilizing real-world data and live signal acquisition. Students will work with EEG, ECG, and EMG signals, learning to apply **signal processing techniques**, **machine learning algorithms**, and **data visualization methods** to extract meaningful information from biological data.

### Environment Setup
To successfully participate in the course, **an appropriate project setup** must be prepared:

**Requirements**
- `Python 3.10+`
- Required packages listed in `/requirements/requirements.txt`

**Setup Guide**
1. **Install Miniconda** (if not already installed)
2. **Prepare Course Environment**
```
conda create -n biosignals python=3.10 jupyter
conda activate biosignals
```
3. **Install Required Packages**
```
conda install numpy scipy matplotlib pandas
conda install -c conda-forge mne scikit-learn
pip install biosppy wfdb
```
4. **Include Jupyter Notebook**
```
conda install --name=base nb conda kernels
```

To test the setup, download and run the files from [`/Week1/code/`](https://github.com/undeMalum/DecodingLifeSignals/tree/main/Week1/code).

For more information, visit [`/Week1/docs`](https://github.com/undeMalum/DecodingLifeSignals/tree/main/Week1/docs).

### Key Learning Objectives
- Master fundamental concepts in biomedical signal processing
- Gain practical experience with Python-based analysis tools
- Develop skills in data visualization and interpretation
- Apply machine learning techniques to biological signals
- Work with both recorded datasets and real-time signal acquisition
- Understand the clinical and research applications of biosignal analysis

<p style="text-align: right;">(<a href="#readme-top">back to top</a>)</p>

### Course Highlights
- Hands-on experience with OpenBCI and EmotiBit hardware
- Real-world project development
- Access to professional-grade biosignal datasets
- Interactive programming sessions
- Small group collaborative projects

<p style="text-align: right;">(<a href="#readme-top">back to top</a>)</p>

### Prerequisites
- Basic Python programming knowledge
- Fundamental understanding of signal processing concepts
- Interest in biomedical applications and healthcare technology

<p style="text-align: right;">(<a href="#readme-top">back to top</a>)</p>

## Tools and Technologies  
- Python and key scientific libraries ([NumPy](https://numpy.org/), [SciPy](https://scipy.org/), [Pandas](https://pandas.pydata.org))  
- Specialized biosignal processing tools ([MNE-Python](https://mne.tools/stable/index.html), [BioSPPy](https://biosppy.readthedocs.io/en/stable/))  
- Machine learning frameworks ([scikit-learn](https://scikit-learn.org/))  
- Data visualization libraries ([Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/))  
- Signal acquisition hardware ([OpenBCI](https://openbci.com/), [EmotiBit](https://www.emotibit.com/))  

Join us in exploring how technology can help us better understand the complex signals our bodies generate, and discover the potential applications in healthcare, research, and beyond.

<p style="text-align: right;">(<a href="#readme-top">back to top</a>)</p>

## Weekly Materials
Each week contains:
- `/docs`: Technical documentation and readings
- `/code`: Python scripts and notebooks
- `/data`: Sample datasets (when applicable)

<p style="text-align: right;">(<a href="#readme-top">back to top</a>)</p>

## Resources
Additional learning materials and references

<p style="text-align: right;">(<a href="#readme-top">back to top</a>)</p>

## Contact
Athanasios Koutras<br>
Associate Professor<br>
[SIPPRE Group](http://www.sippre-group.com)<br>
Electrical & Computer Engineering Department<br>
University of Peloponnese, Greece<br>
koutras[AT]uop.gr<br>

<p style="text-align: right;">(<a href="#readme-top">back to top</a>)</p>
