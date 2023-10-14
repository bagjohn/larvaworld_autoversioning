import hvplot.pandas
import numpy as np
import pandas as pd
import panel as pn
from panel.template import DarkTheme

import larvaworld.lib.model as model

module_class = model.NeuralOscillator
module_attrs = ['input', 'activation', 'output']
title = 'Neural lateral oscillator'
sidebar_width, sidebar_height = 400, 500
widget_kws={'type':pn.widgets.NumberInput, 'width':int(sidebar_width / 2)-20}
args=["dt","tau","n","m","w_ee","w_ce","w_ec","w_cc","input_noise","output_noise"]

def new_class(cls, **kwargs):
    "Creates a new class which overrides parameter defaults."
    return type(type(cls).__name__, (cls,), kwargs)


c = pn.Param(module_class.param,
             expand_button=True,
             default_precedence=3,
             show_name=False,
             widgets={
                 "base_activation": {'type' : pn.widgets.FloatSlider},
                 "activation_range": {'type' : pn.widgets.RangeSlider},
                 **{arg:widget_kws for arg in args}
             })

# Data and Widgets
N = 100
trange = np.arange(N)
A_in = pn.widgets.FloatSlider(name="input", start=-1, end=1, value=0)

p2 = pn.GridBox(*[c.widget(arg) for arg in args],ncols=2)

p1 = pn.Column(
    pn.pane.Markdown(f"### {title}", align='center'),
    A_in,
    c.widget('base_activation'),
    c.widget('activation_range'),
    p2,
    max_width=sidebar_width,
    max_height=sidebar_height
)


# Interactive data pipeline
def module_tester(A_in):
    M = module_class()
    df = pd.DataFrame(columns=module_attrs, index=trange)
    for i in range(N):
        M.step(A_in=A_in)
        df.loc[i] = {k: getattr(M, k) for k in module_attrs}
    df.index *= M.dt
    return df


plot = hvplot.bind(module_tester, A_in).interactive()

template = pn.template.MaterialTemplate(title='Material Dark', theme=DarkTheme, sidebar_width=sidebar_width)
template.sidebar.append(p1)
template.main.append(
    pn.Card(plot.hvplot(min_height=sidebar_height).output().options(xlabel='time (sec)', ylabel='Neural units'),
            title=title)
)
template.servable();

# Run from terminal with : panel serve module_tester.py --show --autoreload
