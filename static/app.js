const form = document.querySelector('#application');
const results = document.querySelector('#results');
const button = document.querySelector('#submit');
let hasResults = false;
function element(tag, text, className) { const node = document.createElement(tag); node.textContent = text; if (className) node.className = className; return node; }
function resetResults() { hasResults = false; results.replaceChildren(element('p','DETAILS CHANGED','eyebrow'),element('h2','Check the updated application'),element('p','The previous assessment belongs to earlier inputs. Check this application again to calculate new results.','muted')); }
form.addEventListener('input', () => { if (hasResults) resetResults(); });
form.addEventListener('change', () => { if (hasResults) resetResults(); });
document.querySelector('#sample').addEventListener('click', () => { form.reset(); document.querySelector('#error').textContent = ''; resetResults(); });
function rawScores(data, target) {
  const score = element('section','','combined-result');
  score.append(element('strong', `${(data.summary.approval_score*100).toFixed(1)}%`,'combined-score'),element('p','Raw average model score','muted'),element('p','This is not a calibrated chance of approval.','fine'));
  target.append(score);
  data.predictions.forEach(model => {
    const card = element('div','','prediction'); card.append(element('strong',model.name),element('p',model.approved?'Historical approval pattern':'Historical decline pattern','raw-outcome'));
    const track = element('div','','track'), bar = element('div','','bar'); bar.style.width = `${model.probability*100}%`; track.append(bar);
    card.append(track,element('small',`${(model.probability*100).toFixed(1)}% raw model score`)); target.append(card);
  });
}
form.addEventListener('submit', async event => {
  event.preventDefault(); button.disabled=true;button.textContent='Checking data support…';results.setAttribute('aria-busy','true');document.querySelector('#error').textContent='';hasResults=false;
  const submitted = JSON.stringify(Object.fromEntries(new FormData(form)));
  results.replaceChildren(element('h2','Checking application…'));
  try {
    const response=await fetch(form.dataset.endpoint || '/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:submitted});
    const data=await response.json();if(!response.ok)throw new Error(data.error||'Assessment failed.');
    if (submitted !== JSON.stringify(Object.fromEntries(new FormData(form)))) throw new Error('Details changed during the check. Submit the updated application again.');
    results.replaceChildren(element('p','02 / ASSESSMENT & DATA CHECKS','eyebrow'));
    results.append(element('h2',data.summary.assessment));
    const checks=data.screening;
    if(checks.requires_review){
      const warning=element('section','','screening-warning');
      warning.append(element('strong','An approval indication is withheld for these inputs.'));
      const list=document.createElement('ul');checks.flags.forEach(flag=>list.append(element('li',flag.message)));warning.append(list);results.append(warning);
    }
    const facts=element('dl','','repayment-facts');
    for(const [label,value] of [[checks.income_label || 'Monthly income equivalent',checks.monthly_income],['Minimum principal / month',checks.minimum_monthly_principal]]){const row=document.createElement('div');const amount=element('dd',value.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2}));amount.append(element('small',LoanLensMoney.toWords(value),'amount-in-words'));row.append(element('dt',label),amount);facts.append(row);}
    results.append(facts,element('p',`${(checks.principal_income_ratio*100).toFixed(1)}% of reported income would go to principal alone. ${checks.note}`,'fine'));
    if(checks.requires_review){const details=document.createElement('details');details.className='raw-score-details';details.append(element('summary','Inspect raw model scores (limited reliability)'));rawScores(data,details);results.append(details);}else{rawScores(data,results);}
    results.append(element('p','Staff decision: Pending review','review-status pending'),element('p','The mean uses equal weights. Historical training patterns do not establish affordability or bank eligibility.','fine'));
    const link=element('a',`Application #${data.application_id} saved · Open staff review →`);link.href=`/applications/${data.application_id}`;results.append(link);hasResults=true;
    results.scrollIntoView({behavior:'smooth',block:'start'});
  } catch(error){document.querySelector('#error').textContent=error.message;results.replaceChildren(element('h2','Assessment unavailable'),element('p',error.message,'muted'));}
  finally{button.disabled=false;button.replaceChildren(document.createTextNode('Check application '),element('span','→'));results.setAttribute('aria-busy','false');}
});

// An in-page loan-term picker avoids browser-native popup rendering issues.
const termPicker = document.querySelector('#loan-term-picker');
if(termPicker){
const termToggle = document.querySelector('#loan-term-button');
const termList = document.querySelector('#loan-term-options');
const termValue = document.querySelector('#loan-term-value');
const termText = document.querySelector('#loan-term-selected');
const termOptions = Array.from(termList.querySelectorAll('[role="option"]'));
function closeTerms(restoreFocus = false) {
  termList.hidden = true;
  termToggle.setAttribute('aria-expanded', 'false');
  if (restoreFocus) termToggle.focus();
}
function openTerms() {
  termList.hidden = false;
  termToggle.setAttribute('aria-expanded', 'true');
  const selected = termOptions.find(option => option.dataset.value === termValue.value);
  if (selected) selected.focus();
}
function syncTerm() {
  termOptions.forEach(option => {
    const selected = option.dataset.value === termValue.value;
    option.setAttribute('aria-selected', String(selected));
    if (selected) termText.textContent = option.textContent.trim();
  });
}
termToggle.addEventListener('click', () => termList.hidden ? openTerms() : closeTerms());
termToggle.addEventListener('keydown', event => {
  if (['ArrowDown', 'ArrowUp'].includes(event.key)) { event.preventDefault(); openTerms(); }
});
termOptions.forEach((option, index) => {
  option.addEventListener('click', () => {
    termValue.value = option.dataset.value;
    syncTerm();
    closeTerms(true);
    termValue.dispatchEvent(new Event('change', {bubbles: true}));
  });
  option.addEventListener('keydown', event => {
    let target;
    if (event.key === 'ArrowDown') target = (index + 1) % termOptions.length;
    if (event.key === 'ArrowUp') target = (index + termOptions.length - 1) % termOptions.length;
    if (event.key === 'Home') target = 0;
    if (event.key === 'End') target = termOptions.length - 1;
    if (target !== undefined) { event.preventDefault(); termOptions[target].focus(); }
    if (event.key === 'Escape') { event.preventDefault(); closeTerms(true); }
    if (event.key === 'Tab') closeTerms();
  });
});
document.addEventListener('click', event => { if (!termPicker.contains(event.target)) closeTerms(); });
form.addEventListener('reset', () => { closeTerms(); setTimeout(() => { termValue.value = termValue.defaultValue; syncTerm(); }, 0); });
syncTerm();

}
