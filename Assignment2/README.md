Main.ipynb: Serves as the pipeline of the project.

DC.py: Diagnostic classifier for the POS-tagging task.

MaskedTask.py: Performs the mask task evaluation.

senreps.py: Makes sentence representatons from embeddings of the different models used in the project. Also retrieves POS-tags for the corresponding experiment.

tagger.py: Makes a parsed file of the given text files in the data folder.

train.py: Creates the matrix W for the transformation of dutch embeddings to english embeddings.

train_pos.py: Performs the POS-tagging task evaluation.

Note: There is no xlingual data and xlingual data dutch in the data folder because of the size of the sentence embeddings. 
