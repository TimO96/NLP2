import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

methods = ['mikolov', 'xing', 'english', 'native']
bidict = {'from':'van', 'it':'het', 'one':'een', 'to':'naar', 'again':'opnieuw'}
labels = {'mikolov':'Mikolov (2013)', 'xing':'Xing (2015)', 'english':'BERT', 'native':'BERTje'}
df = pd.DataFrame(data={'Words':[], 'Predictions':[], 'Method':[], 'Mask':[]})
sns.set(font_scale=2.3)
for method in methods:
    masking = torch.load('masking_score_'+method+'.pt')
    pos = torch.load('pos_task_'+method+'.pt')
    print(method)
    print('masking:')
    for result in masking:
        CE_loss = [loss[0] for loss in result]
        perplexity = [loss[1] for loss in result]
        print(torch.std_mean(torch.tensor(CE_loss)))
        print(torch.std_mean(torch.tensor(perplexity)))

    print('pos:')
    for result in pos:
        print(torch.std_mean(torch.tensor(result)))

    word_dists = torch.load('word_dist_'+method+'.pt')
    if method=='native':
        word_dist = word_dists[0].most_common(6)
        keys = [word[0] for word in word_dist]
        values = [word[1] for word in word_dist]
        word_dist2 = word_dists[2].most_common(6)
        keys2 = [word[0] for word in word_dist2]
        values2 = [word[1] for word in word_dist2]
        
        #data = pd.DataFrame(data={'Words':keys, 'Predictions':values, 'Method':labels[method], 'Mask':'1'})
        data = pd.DataFrame(data={'Words':keys2, 'Predictions':values2, 'Method':labels[method], 'Mask':'50%'})
        #data = pd.concat([data, data2], ignore_index=True)
        df = pd.concat([df, data], ignore_index=True)
    elif method=='xing':
        word_dist = word_dists[0].most_common(5)
        keys = [bidict[word[0]] if word[0] in bidict else word[0] for word in word_dist]
        values = [word[1] for word in word_dist]
        word_dist2 = word_dists[2].most_common(5)
        keys2 = [bidict[word[0]] if word[0] in bidict else word[0] for word in word_dist2]
        values2 = [word[1] for word in word_dist2]

        #data = pd.DataFrame(data={'Words':keys, 'Predictions':values, 'Method':labels[method], 'Mask':'1'})
        data = pd.DataFrame(data={'Words':keys2, 'Predictions':values2, 'Method':labels[method], 'Mask':'50%'})
        #data = pd.concat([data, data2], ignore_index=True)
        df = pd.concat([df, data], ignore_index=True)

indexNames = df[df['Words'] == '[UNK]'].index
df.drop(indexNames, inplace=True)
print(df)

sns.catplot(x="Words", y="Predictions", hue="Method", col='Mask', data=df,
                height=13, kind="bar", legend=False, palette=[sns.color_palette('muted')[i] for i in [1, 3]], dodge=False)

plt.legend(frameon=False, loc='upper right', fontsize=28)

plt.title('Mask = 50%', fontsize=40)
plt.xlabel('Words', fontsize=35)
plt.ylabel('Predictions', fontsize=35)
plt.xticks(fontsize=25)
plt.yticks(fontsize=35)

plt.show()



