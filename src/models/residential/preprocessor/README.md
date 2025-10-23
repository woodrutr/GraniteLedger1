# Residential Preprocessor

### Table of Contents
- [Introduction](#introduction)
- [Prepare Data](#prepare-data)
- [Model Overview](#model-overview)
- [Code Documentation](#code-documentation)

## Introduction

The residential preprocessor builds the baseprice input file used within the residential module. The inputs for the residential module are pre-built within the project. The files in the preprocessor allow users to see how the input files were created and/or allow users to modify input assumptions to explore possible alternative scenarios. The BaseElecPrice.csv file provides an initial set of hourly electricity prices for all regions for the first model year. 

## Prepare Data

If users are interested in reproducing or modifying [BaseElecPrice.csv](/src/input/residential/BaseElecPrice.csv) file, they can run the the Generate Inputs file. 

## Model Overview

This code runs the electricity module and generate hourly electricity prices for all regions for the first model year. It saves those prices within the input directory for the residential module. 

## Code Documentation

[Code Documentation](/docs/README.md)

