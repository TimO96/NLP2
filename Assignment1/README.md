# Probing Language Models

Repository for the NLP2 assignment on probing language models.

## Getting Started:
### POS-tagging task
- To run the POS-tagging task
```bash
python train_pos.py
```
- If you want to run the control task you can set 
```bash
control=True
```
### Structural dependency task
- To run the structural dependency task
```bash
python train.py
```
- If you want to run the control task you can set 
```bash
control=True
```
### Visualizations
- To get visualizations of the UUAS scores with increasing ranks:
```bash
python visualizations.py
```

- Beforehand: set the score files to the saved scores by the dependency task
