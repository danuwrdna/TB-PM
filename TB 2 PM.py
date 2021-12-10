#!/usr/bin/env python
# coding: utf-8

# In[32]:


# mengkoneksikan colab dengan google drive
from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


# Import library
import pandas as pd
import numpy as np
#from datetime import date
from IPython.display import Image
pd.options.mode.chained_assignment = None  # default='warn'


# In[ ]:


get_ipython().run_line_magic('ls', '')


# In[ ]:


# import dataset kedalam program python
event_df = pd.read_csv('running-example.csv', sep=';', low_memory=False)


# In[ ]:


# Melihat Ringkasan dataset
event_df.head()


# In[ ]:


# melihat bentuk pada dataset
print('{} rows and {} columns.'.format(event_df.shape[0], event_df.shape[1]))


# In[ ]:


# melakukan pemilihan column yang digunakan dan membentuk dataset baru
events = event_df[['case_id', 'activity', 'timestamp']]


# In[ ]:


# merubah nama colum yang digunakan pada dataset
events.rename(columns={'case_id':'case',
                       'activity':'label',
                       'timestamp':'datetime'
                       }, inplace=True)


# In[ ]:


# melihat ringkasan dataset
events.head()


# In[ ]:


# melihat type data pada dataset
events.dtypes


# In[ ]:


# merubah type data pada colum case menjadi string (object)
events['case'] = events.astype(str)


# In[ ]:


# merubah type data pada column datetime menjadi datetime
events['datetime'] = pd.to_datetime(events['datetime'])


# In[ ]:


# melihat type data pada dataset
events.dtypes


# In[ ]:


# melakukan pengecekan missing value pada dataset
events.isna().sum()


# In[ ]:


events.head()


# In[ ]:


# Installasi modul PM4PY
get_ipython().system('pip install pm4py')


# In[ ]:


events.rename(columns={'datetime': 'time:timestamp', 'case': 'case:concept:name', 'label': 'concept:name'}, inplace=True)


# In[ ]:


from pm4py.algo.filtering.pandas.start_activities import start_activities_filter
log_start = start_activities_filter.get_start_activities(events)
df_start_activities = start_activities_filter.apply(events, ["register request"])


# In[ ]:


print(events)


# In[ ]:


df_start_activities


# In[ ]:


from pm4py.algo.filtering.pandas.end_activities import end_activities_filter
end_activities = end_activities_filter.get_end_activities(df_start_activities)
filtered_df = end_activities_filter.apply(df_start_activities, ["pay compensation", "reject request"])


# In[ ]:


filtered_df


# In[ ]:


from pm4py.statistics.traces.generic.pandas import case_statistics
variants_count = case_statistics.get_variant_statistics(filtered_df)
variants_count = sorted(variants_count, key=lambda x: x['case:concept:name'], reverse=True)


# In[ ]:


variants_count


# In[ ]:


df = pd.DataFrame(variants_count)


# In[ ]:


df


# In[ ]:


from pm4py.objects.conversion.log import converter as log_converter

# mengkonversi dataset csv kedalam bentuk format log XES
log = log_converter.apply(filtered_df)


# In[ ]:


log


# In[ ]:


from pm4py.algo.discovery.alpha import algorithm as alpha_miner

net, initial_marking, final_marking = alpha_miner.apply(log)


# In[ ]:


from pm4py.visualization.petrinet import visualizer as pn_visualizer

gviz = pn_visualizer.apply(net, initial_marking, final_marking)
pn_visualizer.view(gviz)


# In[ ]:


parameters = {pn_visualizer.Variants.FREQUENCY.value.Parameters.FORMAT: "png"}
gviz = pn_visualizer.apply(net, initial_marking, final_marking, 
                           parameters=parameters, 
                           variant=pn_visualizer.Variants.FREQUENCY, 
                           log=log)

pn_visualizer.view(gviz)


# In[ ]:


from pm4py.algo.discovery.inductive import algorithm as inductive_miner
# Discover process tree using inductive miner
tree = inductive_miner.apply_tree(log)


# In[ ]:


from pm4py.visualization.process_tree import visualizer as pt_visualizer
# Visualise the tree
gviz = pt_visualizer.apply(tree)
pt_visualizer.view(gviz)


# In[ ]:


## Either discover the petri net using inductive miner
net, initial_marking, final_marking = inductive_miner.apply(log)
## Then visualise
gviz = pn_visualizer.apply(net, initial_marking, final_marking, 
                           variant=pn_visualizer.Variants.FREQUENCY, 
                           log=log)
pn_visualizer.view(gviz)


# In[ ]:


## Import heuristics miner algorithm
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner

# heuristics miner
heu_net = heuristics_miner.apply_heu(log)


# In[ ]:


# Import the heuristics net visualisation object
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer
# Visualise model
gviz = hn_visualizer.apply(heu_net)
hn_visualizer.view(gviz)


# In[ ]:


# heuristics miner algorithm returning model, initial marking and
# final marking
net, im, fm = heuristics_miner.apply(log)
# Petri net visualisation
gviz = pn_visualizer.apply(net, im, fm)
pn_visualizer.view(gviz)

