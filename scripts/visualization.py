from matplotlib.colors import LinearSegmentedColormap

colors = [(0, 0, 0), (1, 0, 0)] # first color is black, last is red
cm = LinearSegmentedColormap.from_list(
        "Custom", colors, N=20)

def visualize_attention(attention_score, layer_name):
  
  attention_scores = attention_score.squeeze().cpu().numpy()[:,:15,:15]
  
  n = attention_scores.shape[-1]
  
  labels = ['r', 'ku','kr','kc','s','a','r', 'ku','kr','kc','s','a','r', 'ku','kr']



  fig, ax = plt.subplots()
  im = ax.imshow(np.mean(attention_scores,axis=0),  cmap='inferno')
  
  # plt.xlim([0, n])
  # plt.ylim([0, n])

  ax.set_xlabel("Input Sequence")
  ax.set_ylabel("Output Sequence")

  cbar = ax.figure.colorbar(im, ax=ax)

  ax.set_title(f"Attention Map, layer: {layer_name}")
  ax.set_xticklabels(labels)
  plt.savefig(f"Attention_Map_layer_{layer_name}.pdf")

  plt.show()

