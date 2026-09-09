/** Numerical records shared by the worked visualizations and their tests. */
export function productTrace(inputs, weights, bias=0) {
  if (inputs.length !== weights.length || !Number.isFinite(bias)) throw new Error('Mismatched dot product');
  let sum = 0;
  const terms = Array.from(inputs, (input, index) => {
    const weight = weights[index];
    if (!Number.isFinite(input) || !Number.isFinite(weight)) throw new Error('Non-finite term');
    const product = input * weight;
    const previous = sum;
    sum += product;
    return {index, input, weight, product, previous, sum};
  });
  return {terms, subtotal: sum, bias, total: sum + bias};
}
export function groupedContributions(inputs, weights, count=10) {
  const trace = productTrace(inputs, weights);
  const ranked = [...trace.terms].sort((a,b) => Math.abs(b.product)-Math.abs(a.product));
  const prominent = ranked.slice(0,count);
  const rest = ranked.slice(count);
  return [...prominent.map(term => ({label: `x${term.index}`, value: term.product, index:term.index})),
    ...(rest.length ? [{label:`${rest.length} others`, value:rest.reduce((sum,term)=>sum+term.product,0), index:null}] : [])];
}
export function softmaxTrace(logits) {
  const max = Math.max(...logits);
  const shifted = Array.from(logits, value=>value-max);
  const exponentials = shifted.map(value=>Math.exp(value));
  const denominator = exponentials.reduce((sum,value)=>sum+value,0);
  return {max,shifted,exponentials,denominator,probabilities:exponentials.map(value=>value/denominator)};
}
export function chainForOutputWeight(hidden, probability, weight, isTarget) {
  const outputDerivative = probability - Number(isTarget);
  return {outputDerivative,localWeightDerivative:hidden,weightGradient:outputDerivative*hidden,inputGradient:outputDerivative*weight};
}
