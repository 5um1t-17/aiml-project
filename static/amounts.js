/* Display helpers only: numeric input values and model payloads stay unchanged. */
(() => {
  const ones = 'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split(' ');
  const tens = 'zero ten twenty thirty forty fifty sixty seventy eighty ninety'.split(' ');
  const formatter = new Intl.NumberFormat('en-IN', {useGrouping: false, minimumFractionDigits: 2, maximumFractionDigits: 2});
  function wholeWords(number) {
    if (number < 20) return ones[number];
    if (number < 100) return tens[Math.floor(number / 10)] + (number % 10 ? ' ' + ones[number % 10] : '');
    for (const [scale, label] of [[10000000, 'crore'], [100000, 'lakh'], [1000, 'thousand'], [100, 'hundred']]) {
      if (number >= scale) return wholeWords(Math.floor(number / scale)) + ' ' + label + (number % scale ? ' ' + wholeWords(number % scale) : '');
    }
  }
  function toWords(value) {
    if (value === null || value === undefined || String(value).trim() === '') return '';
    const amount = Number(value);
    if (!Number.isFinite(amount) || Math.abs(amount) > 1000000000000000) return '';
    const [whole, fraction] = formatter.format(Math.abs(amount)).split('.');
    let words = (amount < 0 && (Number(whole) || Number(fraction)) ? 'minus ' : '') + wholeWords(Number(whole));
    if (fraction !== '00') words += ' point ' + [...fraction].map(digit => ones[Number(digit)]).join(' ');
    return words[0].toUpperCase() + words.slice(1);
  }
  window.LoanLensMoney = {toWords};
  document.querySelectorAll('[data-money-input]').forEach((input, index) => {
    const note = document.createElement('small');
    note.className = 'amount-in-words';
    note.id = `money-words-${index}`;
    input.insertAdjacentElement('afterend', note);
    input.setAttribute('aria-describedby', [input.getAttribute('aria-describedby'), note.id].filter(Boolean).join(' '));
    function update() { note.textContent = toWords(input.value); note.hidden = !note.textContent; }
    input.addEventListener('input', update);
    input.addEventListener('change', update);
    input.form?.addEventListener('reset', () => setTimeout(update, 0));
    update();
  });
})();
