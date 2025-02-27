

import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})


DPI = 100


def _symmetry(A, ord='fro'):

    A_sym = 0.5 * (A + A.T)
    A_anti = 0.5 * (A - A.T)

    s_sym = np.linalg.norm(A_sym, ord=ord)
    s_anti = np.linalg.norm(A_anti, ord=ord)

    s = (s_sym - s_anti) / (s_sym + s_anti)

    return s


def _get_filename(method, frac, src, tar):
    return method + '-' + str(frac) + '-' + src + '2' + tar + '.logs'


def _parse_file_contents(contents):
    lines = contents.split('\n')
    mIoU = -1
    total_runtime = 0
    for line in lines:
        if "Elapsed " in line:
            s = line.split(': ')[1].split(', ')
            h = int(s[0].replace(' hours', ''))
            m = int(s[1].replace(' minutes', ''))
            s = float(s[2].replace(' seconds', ''))
            total_runtime += h*3600 + m*60 + s
        if line.startswith('{'):
            mIoU = float(line.split(', ')[1].split(': ')[1][:-1])
            break
    return mIoU, total_runtime


log_dir = '/home/jorisro/research/domain-adaptive-segmentation/train/unsupervised-da/logs'

domains = ['EPFL', 'evhela', 'Kasthuri', 'MiRA', 'MitoEM-H', 'MitoEM-R', 'po936q', 'UroCell', 'VNC']
methods = ['no-da', 'mmd', 'dat', 'ynet', 'unet-ts']
methods_nice = {'no-da': 'No-DA', 'mmd': 'MMD', 'dat': 'DAT', 'ynet': 'Y-Net', 'unet-ts': 'UNet-TS'}
domains_train = {'EPFL': 272*0.4, 'evhela': 360*0.48, 'Kasthuri': 424*0.426, 'MiRA': 2250*0.48, 'MitoEM-H': 16777*0.48,
                 'MitoEM-R': 16777*0.48, 'po936q': 185*0.48, 'UroCell': 85*0.48, 'VNC': 21*0.3}

hmaps = {}
tmaps = {}
mean_to = np.zeros((len(methods), len(domains)))
mean_from = np.zeros((len(methods), len(domains)))
tmean_to = np.zeros((len(methods), len(domains)))
tmean_from = np.zeros((len(methods), len(domains)))
vmin = 1
vmax = 0
for k, method in enumerate(methods):
    hmap = np.zeros((len(domains), len(domains)))
    tmap = np.zeros((len(domains), len(domains)))
    for i, src in enumerate(domains):
        for j, tar in enumerate(domains):
            if src != tar:
                filename = os.path.join(log_dir, _get_filename(method, 0.0, src, tar))
                with open(filename) as f:
                    contents = f.read()
                    mIoU, total_runtime = _parse_file_contents(contents)
                    hmap[i, j] = mIoU
                    tmap[i, j] = total_runtime
            else:
                hmap[i, j] = 0.5
    hmaps[method] = hmap
    tmaps[method] = tmap
    mean_to[k] = hmap.mean(axis=0)
    mean_from[k] = hmap.mean(axis=1)
    tmean_to[k] = tmap.mean(axis=0)
    tmean_from[k] = tmap.mean(axis=1)
    if method != 'no-da':
        vmin = min(vmin, np.min(hmap - hmaps['no-da']))
        vmax = max(vmax, np.max(hmap - hmaps['no-da']))

for k, m in enumerate(methods):
    plt.figure(figsize=(12, 10), dpi=DPI)
    if m == 'no-da':
        sns.heatmap(hmaps[m]*100, xticklabels=domains, yticklabels=domains, cmap='plasma', annot=True)
        plt.title(methods_nice[m] + ' performance', fontsize=22)
    else:
        sns.heatmap((hmaps[m] - hmaps['no-da'])*100, xticklabels=domains, yticklabels=domains, vmin=vmin*100,
                    vmax=vmax*100, cmap='PuOr', center=0, annot=True)


        plt.title(methods_nice[m] + r' $\Delta$' +  ' performance compared to No-DA', fontsize=22)
    plt.savefig('unsupervised-da-%s.pdf' % (m), format='pdf')
    plt.show()

    print('Method: %s' % methods_nice[m])
    print('    Average mIoU: %.2f' % (hmaps[m].mean()*100))
    print('    Degree of symmetry: %.2f' % _symmetry(hmaps[m]))
    print('    Average runtime: %.2f hours' % (tmaps[m].mean() / 3600))

print()
print('Average from/to mIoU: ')
tm = np.sum((mean_from + mean_to) / 2, axis=0) / len(methods)
for i, src in enumerate(domains):
    print('    %s: %.2f' % (src, tm[i]*100))

plt.figure(figsize=(15, 15), dpi=DPI)
for j, domain in enumerate(domains):
    plt.subplot(3, 3, j+1)
    barlist = plt.bar(np.arange(len(methods)), mean_from[:, j])
    barlist[0].set_color('tab:blue')
    barlist[1].set_color('tab:orange')
    barlist[2].set_color('tab:green')
    barlist[3].set_color('tab:red')
    barlist[4].set_color('tab:purple')
    plt.xticks(np.arange(len(methods)), methods, fontsize=12)
    plt.ylim([0.45, 0.62])
    plt.title(domain)
plt.savefig('unsupervised-da-from.pdf', format='pdf')
plt.show()

plt.figure(figsize=(15, 15), dpi=DPI)
for j, domain in enumerate(domains):
    plt.subplot(3, 3, j+1)
    barlist = plt.bar(np.arange(len(methods)), mean_to[:, j])
    barlist[0].set_color('tab:blue')
    barlist[1].set_color('tab:orange')
    barlist[2].set_color('tab:green')
    barlist[3].set_color('tab:red')
    barlist[4].set_color('tab:purple')
    plt.xticks(np.arange(len(methods)), methods, fontsize=12)
    plt.ylim([0.45, 0.62])
    plt.title(domain)
plt.savefig('unsupervised-da-to.pdf', format='pdf')
plt.show()

plt.rcParams.update({'font.size': 22})
plt.figure(figsize=(15, 15), dpi=DPI)
for j, method in enumerate(methods):
    plt.scatter(domains_train.values(), mean_to[j, :])
    plt.xscale('log')
plt.xlabel('Size source training set (megavoxels)')
plt.ylabel('Average mIoU (unsupervised DA)')
plt.legend([methods_nice[method] for method in methods])
plt.savefig('unsupervised-da-to-labels.pdf', format='pdf')
plt.show()
plt.rcParams.update({'font.size': 16})