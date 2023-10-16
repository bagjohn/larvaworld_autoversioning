
import panel as pn
import pandas as pd
import holoviews as hv

from holoviews import opts
from holoviews.streams import Pipe, Buffer

from panel.template import DarkTheme

import larvaworld
import larvaworld.lib.model as model
import larvaworld.lib.reg as reg
import larvaworld.lib.aux as aux

Cmod, Cbody, Ceffector, Csens, Cmem, Cener = 'purple', 'orange', 'red', 'green', 'blue', 'yellow'
Mod_col_dict = {
    'crawler': Ceffector,
    'turner': Ceffector,
    'interference': Cmod,
    'intermitter': Cmod,
    'olfactor': Csens,
    'toucher': Csens,
    'windsensor': Csens,
    'thermosensor': Csens,
    'memory': Cmem,
    'body': Cbody,
    'physics': Cbody,
    'feeder': Ceffector,
    'DEB': Cener,
    'gut': Cener,
    # 'energetics': Cener,
}
ModType_dic = {

    'modulation/memory': ['intermitter', 'interference', 'memory'],
    'effectors': ['crawler', 'turner', 'feeder'],
    'locomotion': ['crawler', 'turner', 'interference','intermitter', 'feeder'],
    # 'effectors': ['crawler', 'turner', 'feeder'],
    'sensors': ['olfactor', 'toucher', 'windsensor', 'thermosensor'],
    # 'modulation/memory': ['intermitter'],
    'body': ['body', 'physics'],
    'energetics': ['DEB', 'gut'],
    # 'energetics': ['energetics'],
}


CT = reg.conf.Model
Msel = pn.widgets.Select(value=CT.dict['forager'],name="larva-model", options=CT.dict, width=100)
Mrun = pn.widgets.Button(name="Run")
Mrow=pn.Row(Msel,Mrun)

B=None


def brain_builder(mconf):
    bconf = mconf.brain
    return model.DefaultBrain(conf=bconf, dt=0.1)

def brain_inspector(B):
    l = []
    for m in ModType_dic['locomotion']:
        obj = getattr(B.locomotor, m)
        if obj:
            _class = obj.__class__
            defaults = larvaworld.lib.param.class_defaults(_class, excluded=[model.Timer])
            args = defaults.keylist
            try:
                c = pn.Card(
                    pn.Param(obj,
                             expand_button=True,
                             default_precedence=3,
                             show_name=False,
                             parameters=args,
                             ),
                    max_width=280, margin=20,
                    header=pn.pane.Markdown(f"### {m} : {_class.name}", align='center'),
                    header_background=Mod_col_dict[m]
                )
                l.append(c)
                # mod_pns[m]=c
            except:
                pass
    # row = pn.GridSpec(objects=mod_pns)
    return pn.Column(*l, max_width=280)


def model_inspector(mconf):
    bconf = mconf.brain
    modIDs = pn.GridBox(*[pn.widgets.Checkbox(name=m, value=v) for m, v in bconf.modules.items()], ncols=1)
    B = brain_builder(mconf)
    l = brain_inspector(B)
    l.insert(0,modIDs)
    return l




attrs = ['x', 'scaled_velocity', 'angular_velocity']
example = pd.DataFrame({a: [] for a in attrs}, columns=attrs)

dfstreams={}
dmaps=[]
for attr in attrs[1:]:
    dfstreams[attr] = Buffer(example[['x',attr]], length=100, index=False)
    dmap = hv.DynamicMap(hv.Curve, streams=[dfstreams[attr]])
    dmap.opts(xlabel='time (sec)', ylabel=attr, width=800, height=200)
    dmaps.append(dmap)

plot = hv.Layout(dmaps).cols(1)
# plot.opts(
    # opts.Points(color='count', line_color='black', size=5, padding=0.1, xaxis=None, yaxis=None),
    # opts.Curve(vdims=['angular_velocity']),
    # xlabel='time (sec)', ylabel='Units')
def trigger_run(v):
    if v :
        # B=_brain(Msel)
        B=brain_builder(Msel.value)
        for i in range(500):
            lin, ang, feed_motion = B.locomotor.step(A_in=0)
            df=pd.DataFrame([(i*B.dt, lin, ang)], columns=attrs)
            for attr in attrs[1:]:
                dfstreams[attr].send(df[['x',attr]])
    return plot

# def run_brain(B):
#     for i in range(500):
#         lin, ang, feed_motion = B.locomotor.step(A_in=0)
#         df=pd.DataFrame([(i*B.dt, lin, ang)], columns=attrs)
#         for attr in attrs[1:]:
#             dfstreams[attr].send(df[['x',attr]])
#     return plot



w, h = 300, 500
w2 = int(w / 2) - 20
template = pn.template.MaterialTemplate(title='Material Dark', theme=DarkTheme, sidebar_width=w)


def _brain(widget):
    return pn.bind(brain_builder, widget)


template.sidebar.append(Mrow)
template.sidebar.append(pn.bind(brain_inspector, _brain(Msel)))
template.main.append(pn.bind(trigger_run, Mrun))
template.servable();
pn.serve(template)

# Run from terminal with : panel serve neural_oscillator_tester.py --show --autoreload
