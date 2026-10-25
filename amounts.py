"""Presentation-only monetary wording, using Indian lakh/crore grouping."""
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

ONES = ('zero one two three four five six seven eight nine ten eleven twelve '
        'thirteen fourteen fifteen sixteen seventeen eighteen nineteen').split()
TENS = 'zero ten twenty thirty forty fifty sixty seventy eighty ninety'.split()


def _whole_words(number):
    if number < 20:
        return ONES[number]
    if number < 100:
        return TENS[number // 10] + (' ' + ONES[number % 10] if number % 10 else '')
    for scale, label in ((10000000, 'crore'), (100000, 'lakh'), (1000, 'thousand'), (100, 'hundred')):
        if number >= scale:
            quotient, remainder = divmod(number, scale)
            return _whole_words(quotient) + ' ' + label + (' ' + _whole_words(remainder) if remainder else '')


def _rounded_amount(value):
    try:
        amount = Decimal(str(value))
        if not amount.is_finite() or abs(amount) > Decimal('1000000000000000'):
            return None
        amount = amount.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError, TypeError):
        return None
    return amount if amount else abs(amount)


def format_amount(value):
    amount = _rounded_amount(value)
    return format(amount, ',.2f') if amount is not None else ''


def amount_in_words(value):
    """Round like a two-decimal amount display; never assume a currency."""
    amount = _rounded_amount(value)
    if amount is None:
        return ''
    whole, fraction = format(abs(amount), '.2f').split('.')
    words = ('minus ' if amount < 0 else '') + _whole_words(int(whole))
    if fraction != '00':
        words += ' point ' + ' '.join(ONES[int(digit)] for digit in fraction)
    return words.capitalize()
