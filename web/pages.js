/** Real static routes. The same filenames are emitted by build and served locally. */
export const PAGES = [
  { id: 'overview', file: 'index.html', title: 'A digit, taken apart.', label: 'The whole network', deck: 'Draw something. Then follow the numbers from the first pixel to the final prediction.' },
  { id: 'input', file: 'input.html', step: 0, title: 'First, an image becomes numbers.', label: 'Pixels & input', deck: 'A handwritten digit is a 28 × 28 matrix. Pick a pixel and change the value the network receives.' },
  { id: 'conv1', file: 'conv1.html', step: 1, title: 'Nine multiplications. One response.', label: 'Convolution I', deck: 'Follow one patch through a filter. Multiply corresponding cells, accumulate the products, then add the bias.' },
  { id: 'relu1', file: 'relu1.html', step: 2, title: 'A simple rule with a sharp corner.', label: 'ReLU I', deck: 'Watch a number pass through max(0, x). Positive responses survive. Negative responses stop here.' },
  { id: 'pool1', file: 'pool1.html', step: 3, title: 'Four values enter. One leaves.', label: 'Max pooling I', deck: 'Compare the four responses in each window and trace the winning value into the smaller map.' },
  { id: 'conv2', file: 'conv2.html', step: 4, title: 'Eight channels, working together.', label: 'Convolution II', deck: 'One filter now spans eight input maps. See how 72 products and one bias become a single activation.' },
  { id: 'relu2', file: 'relu2.html', step: 5, title: 'The same rule, richer patterns.', label: 'ReLU II', deck: 'Apply the nonlinear gate to all sixteen feature maps from the second convolution.' },
  { id: 'pool2', file: 'pool2.html', step: 6, title: 'Keep a smaller record of the evidence.', label: 'Max pooling II', deck: 'Sixteen 14 × 14 maps become sixteen 7 × 7 maps. No weights are learned in this step.' },
  { id: 'flatten', file: 'flatten.html', step: 7, title: 'Unroll the maps, in order.', label: 'Flatten', deck: 'Trace any of the 784 features from its channel, row, and column into a single vector position.' },
  { id: 'dense', file: 'dense.html', step: 8, title: 'What goes into one neuron?', label: 'The dense layer', deck: 'Connect all 784 features to a hidden neuron. Inspect the signed contributions and their running sum.' },
  { id: 'logits', file: 'logits.html', step: 9, title: 'One score for each possible digit.', label: 'Class scores', deck: 'The 128 hidden activations vote through learned weights. Follow their contributions to each of ten logits.' },
  { id: 'softmax', file: 'softmax.html', step: 10, title: 'From ten scores to a distribution.', label: 'Softmax', deck: 'Subtract the maximum, exponentiate, and normalize. Watch the ten probabilities divide up the same whole.' },
  { id: 'learn', file: 'learning.html', step: 11, title: 'Send the error back through the network.', label: 'Learning & gradients', deck: 'Give the image a label. Trace the derivatives backward, then take a real step that changes the weights.' },
  { id: 'guide', file: 'guide.html', title: 'The ideas behind the arithmetic.', label: 'Field notes', deck: 'A reference for channels, receptive fields, weight sharing, and what a prediction can—and cannot—tell you.' },
];
export const pageForStep = step => PAGES.find(page => page.step === step);
export function resolvePage(pathname) {
  const file = pathname.split('/').filter(Boolean).at(-1);
  return PAGES.find(page => page.file === file) || PAGES[0];
}
