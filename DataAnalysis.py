import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib 
from statistics import variance
from scipy import stats

#initialize arrays. 
#The index is the player number (player 0 has age age[0], gender gender[0], etc)
gender      = []
index       = range(0,44)
age         = []
grade       = [] #grade as in 'groep'
groupsize   = [] #2 or 3 player games
playfreq    = [] #frequency of playing board games (1=yearly,2=monthly,3=weekly)
competitive = [] #competitiveness (1-3) higher is more competitive
complexity  = [] #perceived complexity of game (0-4) higher is more complex
turnstrat   = [] #how often did the player have difficulty chosing a move (0-4) higher is more often
exciting    = [] #how exciting 'spannend' was the game (0-4) higher is more exciting
strategy    = [] #how difficult was it to determine a good strategy (0-4) higher is more difficult
fun         = [] #how fun 'leuk' was the game perceived (0-4) higher is more fun
freetime    = [] #how much would the student like to play the game in free time (0-4) higher is more desire
winloss     = [] #whether or not the player won overall
round1      = [] #score in round 1
round2      = [] #score in round 2
round3      = [] #score in round 3
round4      = [] #score in round 4
round5      = [] #score in round 5
round6      = [] #score in round 6

data = [gender,index,age,grade,groupsize,playfreq,competitive,complexity,turnstrat,exciting,strategy,fun,freetime,winloss,round1,round2,round3,round4,round5,round6]
 
#open and read datafile, fill the data arrays with data
with open('pilot123numbers3.csv') as raw: #read raw data where each row represents a csv file line
    next(raw)                          #skip the first row (column headers)
    
    for row in raw:                    #iterate over the rows. Each row represents one student
        splitrow = row.split(",")      #split row string into an array of smaller strings      
        data[0].append(splitrow[0])    #append age
        
        for i in range (2,19):         #for the other elements, append as int if there is data
            if i == 13:
                data[i].append(splitrow[i]) 
            elif splitrow[i].isdigit():
                data[i].append(int(splitrow[i]))
            else:
                data[i].append(None)
        if splitrow[19][:-1].isdigit():#for the last element, remove linebreak character
            data[19].append(int(splitrow[19][:-1]))
        else:
            data[19].append(None)
            
def printData():     
    for datum in data:
        print(datum)
    
#returns a list of indices of players that are below average in competitiveness 
#and a list of players who are above average in competitiveness
def divideByCompetitiveness():
    compFiltered = list(filter(lambda x: x != None, competitive))
    competitiveSorted = sorted(compFiltered)
    average = competitiveSorted[int(len(competitiveSorted)/2)]
    print(average)
    low = []
    high = []
    
    for i in index:
        if (isinstance(competitive[i], int)):
            if competitive[i] >= average:
                high.append(i)
            else:
                low.append(i)
    return low,high

#returns a list of indices of players that are below average in competitiveness 
#and a list of players who are above average in competitiveness
def divideByAge():
    ageF = list(filter(lambda x: x != None, age))
    ageS = sorted(ageF)
    average = ageS[int(len(ageS)/2)]
    
    low = []
    high = []
    
    for i in index:
        if (isinstance(age[i], int)):
            if age[i] > average:
                high.append(i)
            else:
                low.append(i)
    return low,high

def divideByComplexity():
    compf = list(filter(lambda x: x != None, complexity))
    comps = sorted(compf)
    average = comps[int(len(comps)/2)]
    
    low = []
    high = []
    
    for i in index:
        if (isinstance(complexity[i], int)):
            if complexity[i] > average:
                high.append(i)
            else:
                low.append(i)
    return low,high

def divideByGender():    
    m = []
    f = []
    
    for i in index:
        if gender[i] == 'm':
            m.append(i)
        else:
            f.append(i)
    return m,f



#total points scored over all rounds by player i
def totalScore(i):
    total = 0
    for score in [round1[i],round2[i],round3[i],round4[i],round5[i],round6[i]]:
        if isinstance(score,int):
            total = total + score
    return total


low,high = divideByCompetitiveness()
lowCompScores = map(totalScore,low)
highCompScores = map(totalScore,high)

from matplotlib.pyplot import figure
figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')

#matplotlib.rc('font', **font)

filteredComp = clean = [x for x in competitive if x != None]
sns.set(font_scale=2) 
ax = sns.distplot(list(highCompScores), bins=4, label='High competitiveness group');
ax.set(xlabel='total score')
ax.set(ylabel='fraction')
sns.distplot(list(lowCompScores), bins=4, label='Low competitiveness group');
plt.legend()

low,high = divideByCompetitiveness()
lowCompScores = list(map(totalScore,low))
highCompScores = list(map(totalScore,high))
highwinloss = []
lowwinloss = []
highcount = 0
lowcount = 0


for i in low:
    lowwinloss.append(winloss[i])
    if winloss[i] == 'w':
        lowcount = lowcount + 1
    
for i in high:
    highwinloss.append(winloss[i])
    if winloss[i] == 'w':
        highcount = highcount + 1

lownrplayers = 0        
for i in low:
    lownrplayers = lownrplayers + groupsize[i]
    
highnrplayers = 0        
for i in high:
    highnrplayers = highnrplayers + groupsize[i]
    
print('average group size of high competitiveness players:', highnrplayers/len(high))
print('average group size of low competitiveness players', lownrplayers/len(low))
print('average wins per high competitiveness player', highcount/len(high))
print('average wins per low competitiveness player', lowcount/len(low))
print('average score by low competitiveness players:',sum(lowCompScores)/len(low))
print('average score by high competitiveness players:',sum(highCompScores)/len(high))


print('total wins of high competitiveness player', highcount)
print('total wins of low competitiveness player', lowcount)
print('total losses of high competitiveness player', len(high)-highcount)
print('total losses of low competitiveness player', len(low)-lowcount)

print(variance(lowCompScores))
print(variance(highCompScores))

low,high = divideByCompetitiveness()


def funScore(i):
    return fun[i] + freetime[i]

lowFunScores = list(map(funScore,low))
highFunScores = list(map(funScore,high))

figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
sns.set(font_scale=2) 
ax = sns.distplot(list(highFunScores), bins=4, label='Above average competitiveness');
ax.set(xlabel='total score')
sns.distplot(list(lowFunScores), bins=4, label='Below average competitiveness');
plt.legend()


print('average enjoyment of low comp group:',sum(lowFunScores)/len(low))
print('average enjoyment of high comp group:',sum(highFunScores)/len(high))

highwin  = list(filter(lambda x: winloss[x] == 'w', high))
highloss = list(filter(lambda x: winloss[x] == 'v', high))
lowwin   = list(filter(lambda x: winloss[x] == 'w', low))
lowloss  = list(filter(lambda x: winloss[x] == 'v', low))

print('average enjoyment of high comp winners',sum(list(map(funScore,highwin)))/len(highwin), 'N:',
     len(highwin))
print('average enjoyment of high comp losers',sum(list(map(funScore,highloss)))/len(highloss), 'N:',
     len(highloss))
print('average enjoyment of low comp winners',sum(list(map(funScore,lowwin)))/len(lowwin), 'N:',
     len(lowwin))
print('average enjoyment of low comp losers',sum(list(map(funScore,lowloss)))/len(lowloss), 'N:',
     len(lowloss))

stat,p = stats.mannwhitneyu(list(map(funScore,highwin)),list(map(funScore,highloss)),alternative='two-sided')
if p < a:
    print("The score samples are likely from different distributions, p:",p)
else:
    print("The score sample are likely from the same distribution, p:",p)
stat,p = stats.mannwhitneyu(list(map(funScore,lowwin)),list(map(funScore,lowloss)),alternative='two-sided')
if p < a:
    print("The score samples are likely from different distributions, p:",p)
else:
    print("The score sample are likely from the same distribution, p:",p)
    

from scipy.stats import chisquare
 
X = [11,8,15,9]

exp = [11.5,7.5,14.5,9.5]
 
stat,p = chisquare(f_obs=X, f_exp=exp)
print(p.round(5))

#age things

print(age)
low,high = divideByAge()

lowComplexity = list(map(lambda x: complexity[x],low))
highComplexity = list(map(lambda x: complexity[x],high))

print(sum(lowComplexity)/len(lowComplexity))
print(sum(highComplexity)/len(highComplexity))


from matplotlib.pyplot import figure
figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')

plt.figure(figsize=(10.0,10.0))
plt.xlabel('Complexity (high age group).')
plt.ylabel('Frequency')
plt.xticks(np.arange(0,9))
plt.yticks(np.arange(0,30))
plt.bar(np.arange(0, 5), np.bincount(highComplexity, minlength=5)[0:5])
axes = plt.gca()
axes.set_ylim([0,10])
plt.show()

plt.figure(figsize=(10.0,10.0))
plt.xlabel('Complexity (low age group).')
plt.ylabel('Frequency')
plt.xticks(np.arange(0,9))
plt.yticks(np.arange(0,30))
plt.bar(np.arange(0, 5), np.bincount(lowComplexity, minlength=5)[0:5])
axes = plt.gca()
axes.set_ylim([0,15])
plt.show()

#wilcoxon ranksum test for score of different competitiveness groups
lowComplexity.sort()
highComplexity.sort()
a = 0.05
stat,p = stats.mannwhitneyu(lowComplexity,highComplexity,use_continuity=True,alternative='two-sided')
if p < a:
    print("The score samples are likely from different distributions, p:",p)
else:
    print("The score sample are likely from the same distribution, p:",p)
    

lowExciting = list(map(lambda x: exciting[x],low))
highExcititing = list(map(lambda x: exciting[x],high))

print(sum(lowExciting)/len(lowExciting))
print(sum(highExcititing)/len(highExcititing))


plt.figure(figsize=(9.0,9.0))
plt.xlabel('Excitement (high age group).')
plt.ylabel('Frequency')
plt.xticks(np.arange(0,9))
plt.yticks(np.arange(0,30))
plt.bar(np.arange(0, 5), np.bincount(highExcititing, minlength=5)[0:5])
axes = plt.gca()
axes.set_ylim([0,10])
plt.show()

plt.figure(figsize=(9.0,9.0))
plt.xlabel('Excitement (low age group).')
plt.ylabel('Frequency')
plt.xticks(np.arange(0,9))
plt.yticks(np.arange(0,30))
plt.bar(np.arange(0, 5), np.bincount(lowExciting, minlength=5)[0:5])
axes = plt.gca()
axes.set_ylim([0,15])
plt.show()

#wilcoxon ranksum test for score of different competitiveness groups
lowExciting.sort()
highExcititing.sort()
a = 0.05
stat,p = stats.mannwhitneyu(lowExciting,highExcititing,use_continuity=True,alternative='two-sided')
if p < a:
    print("The score samples are likely from different distributions, p:",p)
else:
    print("The score sample are likely from the same distribution, p:",p)
    
print(len(low), len(high))

def totalStrategy(i):
    return turnstrat[i]+strategy[i]


lowStrat = list(map(lambda x: totalStrategy(x),low))
highStrat = list(map(lambda x: totalStrategy(x),high))

print(sum(lowStrat)/len(lowStrat))
print(sum(highStrat)/len(highStrat))

plt.figure(figsize=(9.0,9.0))
plt.xlabel('Strategy (high age group).')
plt.ylabel('Frequency')
plt.xticks(np.arange(0,9))
plt.yticks(np.arange(0,30))
plt.bar(np.arange(0, 9), np.bincount(highStrat, minlength=9)[0:9])
axes = plt.gca()
axes.set_ylim([0,10])
plt.show()

plt.figure(figsize=(9.0,9.0))
plt.xlabel('Strategy (low age group).')
plt.ylabel('Frequency')
plt.xticks(np.arange(0,9))
plt.yticks(np.arange(0,30))
plt.bar(np.arange(0, 9), np.bincount(lowStrat, minlength=9)[0:9])
axes = plt.gca()
axes.set_ylim([0,15])
plt.show()
#wilcoxon ranksum test for score of different competitiveness groups
lowStrat.sort()
highExcititing.sort()
a = 0.05
stat,p = stats.mannwhitneyu(lowStrat,highStrat,use_continuity=True,alternative='two-sided')
if p < a:
    print("The score samples are likely from different distributions, p:",p)
else:
    print("The score sample are likely from the same distribution, p:",p)
    
print(len(low), len(high))

m,f = divideByGender()

mStrat = list(map(lambda x: totalStrategy(x),m))
fStrat = list(map(lambda x: totalStrategy(x),f))

mScore = list(map(lambda x: totalScore(x),m))
fScore = list(map(lambda x: totalScore(x),f))

mCount = 0
fCount = 0

for i in f:
    if winloss[i] == 'w':
        fCount = fCount + 1
    
for i in m:
    if winloss[i] == 'w':
        mCount = mCount + 1

print('m wins', mCount)
print('f wins', fCount)

mEnjoy = list(map(lambda x: fun[x] + freetime[x],m))
fEnjoy = list(map(lambda x:fun[x] + freetime[x],f))

mExcite = list(map(lambda x: exciting[x],m))
fExcite = list(map(lambda x: exciting[x],f))

mComp = list(map(lambda x: complexity[x],m))
fComp = list(map(lambda x: complexity[x],f))

print('mComp', sum(mComp)/len(mComp))
print('fComp', sum(fComp)/len(fComp))

mScore.sort()
fScore.sort()
a = 0.05
stat,p = stats.mannwhitneyu(mScore,fScore,use_continuity=True,alternative='two-sided')
if p < a:
    print("The score samples are likely from different distributions, p:",p)
else:
    print("The score sample are likely from the same distribution, p:",p)
    
mComp.sort()
fComp.sort()
a = 0.05
stat,p = stats.mannwhitneyu(mComp,fComp,use_continuity=True,alternative='two-sided')
if p < a:
    print("The comp samples are likely from different distributions, p:",p)
else:
    print("The comp sample are likely from the same distribution, p:",p)
    
mStrat.sort()
fStrat.sort()
a = 0.05
stat,p = stats.mannwhitneyu(mStrat,fStrat,use_continuity=True,alternative='two-sided')
if p < a:
    print("The strat samples are likely from different distributions, p:",p)
else:
    print("The strat sample are likely from the same distribution, p:",p)
    
mEnjoy.sort()
fEnjoy.sort()
a = 0.05
stat,p = stats.mannwhitneyu(mEnjoy,fEnjoy,use_continuity=True,alternative='two-sided')
if p < a:
    print("The enjoy samples are likely from different distributions, p:",p)
else:
    print("The enjoy sample are likely from the same distribution, p:",p)
    
mExcite.sort()
fExcite.sort()
a = 0.05
stat,p = stats.mannwhitneyu(mExcite,fExcite,use_continuity=True,alternative='two-sided')
if p < a:
    print("The excite samples are likely from different distributions, p:",p)
else:
    print("The excite sample are likely from the same distribution, p:",p)

plt.figure(figsize=(9.0,9.0))    
plt.xlabel('Strategic difficulty (male students).')
plt.ylabel('Frequency')

plt.xticks(np.arange(0,10))
plt.yticks(np.arange(0,30))
plt.bar(np.arange(0, 9), np.bincount(mStrat, minlength=9)[0:9])
plt.show()

plt.figure(figsize=(9.0,9.0))
plt.xlabel('Strategic difficulty (female students).')
plt.ylabel('Frequency')
plt.xticks(np.arange(0,10))
plt.yticks(np.arange(0,30))
plt.bar(np.arange(0, 9), np.bincount(fStrat, minlength=9)[0:9])
plt.show()

plt.figure(figsize=(9.0,9.0))
plt.xlabel('Enjoyment (male students).')
plt.ylabel('Frequency')
plt.xticks(np.arange(0,10))
plt.yticks(np.arange(0,30))
plt.bar(np.arange(0, 9), np.bincount(mEnjoy, minlength=9)[0:9])
plt.show()

plt.figure(figsize=(9.0,9.0))
plt.xlabel('Enjoyment (female students).')
plt.ylabel('Frequency')
plt.xticks(np.arange(0,10))
plt.yticks(np.arange(0,30))
plt.bar(np.arange(0, 9), np.bincount(fEnjoy, minlength=9)[0:9])
plt.show()

plt.figure(figsize=(9.0,9.0))
plt.xlabel('Excitement (male students).')
plt.ylabel('Frequency')
plt.xticks(np.arange(0,10))
plt.yticks(np.arange(0,30))
plt.bar(np.arange(0, 5), np.bincount(mExcite, minlength=5)[0:5])
plt.show()

plt.figure(figsize=(9.0,9.0))
plt.xlabel('Excitement (female students).')
plt.ylabel('Frequency')
plt.xticks(np.arange(0,10))
plt.yticks(np.arange(0,30))
plt.bar(np.arange(0, 5), np.bincount(fExcite, minlength=5)[0:5])
plt.show()

plt.figure(figsize=(9.0,9.0))
plt.xlabel('Complexity (female students).')
plt.ylabel('Frequency')
plt.xticks(np.arange(0,10))
plt.yticks(np.arange(0,30))
plt.bar(np.arange(0,5), np.bincount(fComp, minlength=5)[0:5])
plt.show()

plt.figure(figsize=(9.0,9.0))
plt.xlabel('Complexity (male students).')
plt.ylabel('Frequency')
plt.xticks(np.arange(0,10))
plt.yticks(np.arange(0,30))
plt.bar(np.arange(0,5), np.bincount(mComp, minlength=5)[0:5])
plt.show()



low,high = divideByComplexity()

lowFun = list(map(lambda x: fun[x]+freetime[x], low))
highFun = list(map(lambda x: fun[x]+freetime[x], high))


plt.figure(figsize=(9.0,9.0))
plt.xlabel('Enjoyment (high complexity group).')
plt.ylabel('Frequency')
plt.xticks(np.arange(0,10))
plt.yticks(np.arange(0,30))
plt.bar(np.arange(0,9), np.bincount(highFun, minlength=9)[0:9])
plt.show()


plt.figure(figsize=(9.0,9.0))
plt.xlabel('Enjoyment (low complexity group).')
plt.ylabel('Frequency')
plt.xticks(np.arange(0,10))
plt.yticks(np.arange(0,30))
plt.bar(np.arange(0,9), np.bincount(lowFun, minlength=9)[0:9])
plt.show()

a = 0.05
stat,p = stats.mannwhitneyu(highFun,lowFun,use_continuity=True,alternative='two-sided')
if p < a:
    print("The fun samples are likely from different distributions, p:",p)
else:
    print("The fun sample are likely from the same distribution, p:",p)
    
i = 0
groupnrs = []
groupnr = 1
while i < 44:
    if groupsize[i] == 2:
        groupnrs += [groupnr,groupnr]
        groupnr = groupnr + 1
        i = i + 2
    else:
        groupnrs += [groupnr,groupnr,groupnr]
        i = i + 3
        groupnr = groupnr + 1
        
#Returns how close the game was. 
#returns the average difference in score with each other player in the group
def closeness(index):
    groupnr = groupnrs[index] #get the group number
    group = [i for i, x in enumerate(groupnrs) if x == groupnr] #get all player indices of group
    groupscores = list(map(lambda x: totalScore(x), group)) #get scores for all players in group
    Igroupscores = list(map(lambda x, y:(x,y), group, groupscores)) #zip player indices and total scores
    IFgroupscores = list(filter(lambda x:x[0]!=index, Igroupscores)) #remove our player
    totalDif = 0
    playerScore = totalScore(index)
    for p in IFgroupscores:
        totalDif = totalDif + abs(playerScore - p[1])
    return totalDif / len(IFgroupscores)

close = list(map(lambda x: closeness(x), range(0,44)))

def divideByCloseness():
    closeSorted = sorted(close)
    average = closeSorted[int(len(closeSorted)/2)]
    
    low = []
    high = []
    
    for i in index:
        if close[i] > average:
            high.append(i)
        else:
            low.append(i)
    return low,high

low,high = divideByCloseness()

lowFun = list(map(lambda x: fun[x]+freetime[x], low))
highFun = list(map(lambda x: fun[x]+freetime[x], high))

plt.figure(figsize=(9.0,9.0))
plt.xlabel('Fun (not close games group).')
plt.ylabel('Frequency')
plt.xticks(np.arange(0,9))
plt.yticks(np.arange(0,30))
plt.bar(np.arange(0, 9), np.bincount(highFun, minlength=9)[0:9])
axes = plt.gca()
axes.set_ylim([0,20])
plt.show()

plt.figure(figsize=(9.0,9.0))
plt.xlabel('Fun (very close games group).')
plt.ylabel('Frequency')
plt.xticks(np.arange(0,9))
plt.yticks(np.arange(0,30))
plt.bar(np.arange(0, 9), np.bincount(lowFun, minlength=9)[0:9])
axes = plt.gca()
axes.set_ylim([0,20])
plt.show()

a = 0.05
stat,p = stats.mannwhitneyu(highFun,lowFun,use_continuity=True,alternative='two-sided')
if p < a:
    print("The fun samples are likely from different distributions, p:",p)
else:
    print("The fun sample are likely from the same distribution, p:",p)
    

lowExcite = list(map(lambda x: exciting[x], low))
highExcite = list(map(lambda x: exciting[x], high))

plt.figure(figsize=(9.0,9.0))
plt.xlabel('Excitement (not close games group).')
plt.ylabel('Frequency')
plt.xticks(np.arange(0,9))
plt.yticks(np.arange(0,20))
plt.bar(np.arange(0, 5), np.bincount(highExcite, minlength=5)[0:5])
axes = plt.gca()
axes.set_ylim([0,10])
plt.show()

plt.figure(figsize=(9.0,9.0))
plt.xlabel('Excitement (very close games group).')
plt.ylabel('Frequency')
plt.xticks(np.arange(0,9))
plt.yticks(np.arange(0,30))
plt.bar(np.arange(0, 5), np.bincount(lowExcite, minlength=5)[0:5])
axes = plt.gca()
axes.set_ylim([0,10])
plt.show()

a = 0.05
stat,p = stats.mannwhitneyu(highExcite,lowExcite,use_continuity=True,alternative='two-sided')
if p < a:
    print("The fun samples are likely from different distributions, p:",p)
else:
    print("The fun sample are likely from the same distribution, p:",p)
    
def divideByScore():
    totalScores = list(map(lambda x: totalScore(x), index))
    scoresf = list(filter(lambda x: x != None, totalScores))
    scoress = sorted(scoresf)
    average = scoress[int(len(scoress)/2)]
    low = []
    high = []
    
    for i in index:
        if (isinstance(totalScores[i], int)):
            if totalScores[i] > average:
                high.append(i)
            else:
                low.append(i)
    return low,high


low,high = divideByScore()


lowFun = list(map(lambda x: fun[x]+freetime[x], low))
highFun = list(map(lambda x: fun[x]+freetime[x], high))

plt.figure(figsize=(9.0,9.0))
plt.xlabel('Enjoyment (high scores group).')
plt.ylabel('Frequency')
plt.xticks(np.arange(0,9))
plt.yticks(np.arange(0,30))
plt.bar(np.arange(0, 9), np.bincount(highFun, minlength=9)[0:9])
axes = plt.gca()
plt.show()

plt.figure(figsize=(9.0,9.0))
plt.xlabel('Enjoyment (low scores group).')
plt.ylabel('Frequency')
plt.xticks(np.arange(0,9))
plt.yticks(np.arange(0,30))
plt.bar(np.arange(0, 9), np.bincount(lowFun, minlength=9)[0:9])
axes = plt.gca()
plt.show()

a = 0.05
stat,p = stats.mannwhitneyu(highFun,lowFun,use_continuity=True,alternative='two-sided')
if p < a:
    print("The fun samples are likely from different distributions, p:",p)
else:
    print("The fun sample are likely from the same distribution, p:",p)
    
def divideByWin():
    win = []
    loss = []
    for i in index:
        if winloss[i] == 'w':
            win.append(i)
        else:
            loss.append(i)
    return win,loss
            
win,loss = divideByWin();


lowFun = list(map(lambda x: fun[x]+freetime[x], loss))
highFun = list(map(lambda x: fun[x]+freetime[x], win))

plt.figure(figsize=(9.0,9.0))
plt.xlabel('Enjoyment (winners group).')
plt.ylabel('Frequency')
plt.xticks(np.arange(0,9))
plt.yticks(np.arange(0,30))
plt.bar(np.arange(0, 9), np.bincount(highFun, minlength=9)[0:9])
axes = plt.gca()
plt.show()

plt.figure(figsize=(9.0,9.0))
plt.xlabel('Enjoyment (losers group).')
plt.ylabel('Frequency')
plt.xticks(np.arange(0,9))
plt.yticks(np.arange(0,30))
plt.bar(np.arange(0, 9), np.bincount(lowFun, minlength=9)[0:9])
axes = plt.gca()
plt.show()

a = 0.05
stat,p = stats.mannwhitneyu(highFun,lowFun,use_continuity=True,alternative='two-sided')
if p < a:
    print("The fun samples are likely from different distributions, p:",p)
else:
    print("The fun sample are likely from the same distribution, p:",p)
    
def divideByFreq():
    freqf = list(filter(lambda x: x != None, playfreq))
    freqs = sorted(freqf)
    average = freqs[int(len(freqs)/2)]
    low = []
    high = []
    
    for i in index:
        if (isinstance(playfreq[i], int)):
            if playfreq[i] > average:
                high.append(i)
            else:
                low.append(i)
    return low,high

low,high = divideByFreq();


lowCompl = list(map(lambda x: complexity[x], low))
highCompl = list(map(lambda x: complexity[x], high))

plt.figure(figsize=(12.0,12.0))
plt.xlabel('Complexity (frequent board game players).')
plt.ylabel('Frequency')
plt.xticks(np.arange(0,9))
plt.yticks(np.arange(0,30))
plt.bar(np.arange(0, 5), np.bincount(highCompl, minlength=5)[0:5])
axes = plt.gca()
plt.show()

plt.figure(figsize=(12.0,12.0))
plt.xlabel('Complexity (non frequent board game players).')
plt.ylabel('Frequency')
plt.xticks(np.arange(0,9))
plt.yticks(np.arange(0,30))
plt.bar(np.arange(0, 5), np.bincount(lowCompl, minlength=5)[0:5])
axes = plt.gca()
plt.show()

a = 0.05
stat,p = stats.mannwhitneyu(lowCompl,highCompl,use_continuity=True,alternative='two-sided')
if p < a:
    print("The fun samples are likely from different distributions, p:",p)
else:
    print("The fun sample are likely from the same distribution, p:",p)
    
def divideByGroupSize():
    low = []
    high = []
    for i in index:
        if groupsize[i] == 3:
            high.append(i)
        elif groupsize[i] == 2:
            low.append(i)
        else:
            print('error invalid groupsize')
    return low,high

low,high = divideByGroupSize();

lowCompl = list(map(lambda x: complexity[x], low))
highCompl = list(map(lambda x: complexity[x], high))

lowStrat = list(map(lambda x: totalStrategy(x), low))
highStrat = list(map(lambda x: totalStrategy(x), high))

lowEnjoy = list(map(lambda x: fun[x]+freetime[x], low))
highEnjoy = list(map(lambda x: fun[x]+freetime[x], high))

lowExcite = list(map(lambda x: exciting[x], low))
highExcite = list(map(lambda x: exciting[x], high))


plt.figure(figsize=(12.0,12.0))
plt.xlabel('Experienced complexity (group size 3).')
plt.ylabel('Frequency')
plt.xticks(np.arange(0,9))
plt.yticks(np.arange(0,30))
plt.bar(np.arange(0, 5), np.bincount(highCompl, minlength=5)[0:5])
axes = plt.gca()
plt.show()

plt.figure(figsize=(12.0,12.0))
plt.xlabel('Experienced complexity (group size 2).')
plt.ylabel('Frequency')
plt.xticks(np.arange(0,9))
plt.yticks(np.arange(0,30))
plt.bar(np.arange(0, 5), np.bincount(lowCompl, minlength=5)[0:5])
axes = plt.gca()
plt.show()

a = 0.05
stat,p = stats.mannwhitneyu(lowCompl,highCompl,use_continuity=True,alternative='two-sided')
if p < a:
    print("The fun samples are likely from different distributions, p:",p)
else:
    print("The fun sample are likely from the same distribution, p:",p)
    
plt.figure(figsize=(12.0,12.0))
plt.xlabel('Experienced strategic difficulty (group size 3).')
plt.ylabel('Frequency')
plt.xticks(np.arange(0,9))
plt.yticks(np.arange(0,30))
plt.bar(np.arange(0, 5), np.bincount(highStrat, minlength=5)[0:5])
axes = plt.gca()
plt.show()

plt.figure(figsize=(12.0,12.0))
plt.xlabel('Experienced strategic difficulty (group size 2).')
plt.ylabel('Frequency')
plt.xticks(np.arange(0,9))
plt.yticks(np.arange(0,30))
plt.bar(np.arange(0, 5), np.bincount(lowStrat, minlength=5)[0:5])
axes = plt.gca()
plt.show()

a = 0.05
stat,p = stats.mannwhitneyu(lowStrat,highStrat,use_continuity=True,alternative='two-sided')
if p < a:
    print("The fun samples are likely from different distributions, p:",p)
else:
    print("The fun sample are likely from the same distribution, p:",p)
    
plt.figure(figsize=(12.0,12.0))
plt.xlabel('Experienced enjoyment (group size 3).')
plt.ylabel('Frequency')
plt.xticks(np.arange(0,9))
plt.yticks(np.arange(0,30))
plt.bar(np.arange(0, 9), np.bincount(highEnjoy, minlength=9)[0:9])
axes = plt.gca()
plt.show()

plt.figure(figsize=(12.0,12.0))
plt.xlabel('Experienced enjoyment (group size 2).')
plt.ylabel('Frequency')
plt.xticks(np.arange(0,9))
plt.yticks(np.arange(0,30))
plt.bar(np.arange(0, 9), np.bincount(lowEnjoy, minlength=9)[0:9])
axes = plt.gca()
plt.show()

a = 0.05
stat,p = stats.mannwhitneyu(lowEnjoy,highEnjoy,use_continuity=True,alternative='two-sided')
if p < a:
    print("The fun samples are likely from different distributions, p:",p)
else:
    print("The fun sample are likely from the same distribution, p:",p)
    
plt.figure(figsize=(12.0,12.0))
plt.xlabel('Experienced excitement (group size 3).')
plt.ylabel('Frequency')
plt.xticks(np.arange(0,9))
plt.yticks(np.arange(0,30))
plt.bar(np.arange(0, 5), np.bincount(highExcite, minlength=5)[0:5])
axes = plt.gca()
plt.show()

plt.figure(figsize=(12.0,12.0))
plt.xlabel('Experienced excitement (group size 2).')
plt.ylabel('Frequency')
plt.xticks(np.arange(0,9))
plt.yticks(np.arange(0,30))
plt.bar(np.arange(0, 5), np.bincount(lowExcite, minlength=5)[0:5])
axes = plt.gca()
plt.show()

a = 0.05
stat,p = stats.mannwhitneyu(lowExcite,highExcite,use_continuity=True,alternative='two-sided')
if p < a:
    print("The fun samples are likely from different distributions, p:",p)
else:
    print("The fun sample are likely from the same distribution, p:",p)
