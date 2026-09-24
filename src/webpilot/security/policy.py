"""The security gate: *what may the agent do without asking a human?*

Design
------
The policy is a **rule engine over meaning**, never over sites.  A rule is a
(regex, level, reason) triple that is evaluated against a small set of *signals*
extracted from the tool call and the page context:

``label``       accessible name of the target element (button/link text)
``placeholder`` the field's placeholder
``intent``      the model's short description of the action (a tool argument)
``field_type``  ``input[type=...]`` (``password``, ``file``, ``card``, ...)
``role``        element role from the page sniffer (``button``, ``link``, ...)
``href``/``url``  absolute URLs
``title``       the page title
``text``        ``label + placeholder + intent`` (element-level wording)

Page-level wording (title/path) is deliberately **not** part of ``text``: a page
titled "Delete account" must not turn every harmless click on it into a
destructive action.  Only rules that explicitly opt into the page signals (or
into the derived ``flow``: *payment* / *booking* / *cart*) may use them.

Levels
------
``safe``        nothing to see.
``caution``     mildly irreversible / public / state changing: the UI shows a
                warning, no prompt.  The optional LLM second opinion may
                escalate a caution to destructive.
``destructive`` hard to undo, moves money, publishes on the human's behalf,
                destroys data, or touches credentials: needs confirmation.

Because the rules work on *meaning*, they fire on a site the author has never
seen (see ``test_policy_flags_unseen_checkout_flow_by_meaning``), and they carry
Russian and English wording.  Adding a site to ``auto_approve_domains`` is the
only per-site knob, and it is an explicit human decision.

Every decision goes through :meth:`SecurityPolicy.authorize`, which appends one
JSON line to ``config.audit_path``.  Passwords, card numbers, tokens and other
secret-looking values are redacted **before** they reach the log.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from ..config import Config
from ..types import (
    ActionContext,
    ActionRisk,
    ConfirmFn,
    Element,
    LLMMessage,
    RiskLevel,
    ToolCall,
)

# --------------------------------------------------------------------------- #
# Secrets: never let a credential reach the log, the terminal or an audit file
# --------------------------------------------------------------------------- #

REDACTED = "***REDACTED***"

#: Argument names whose *value* is always replaced, whatever it looks like.
SECRET_KEY_RE = re.compile(
    r"(pass|pwd|secret|token|otp|\bpin\b|pin_?code|cvv|cvc|card|credit|iban|"
    r"ssn|api[-_]?key|auth|credential|passphrase)",
    re.IGNORECASE,
)

#: Values that *look* like a secret even under an innocent key.
SECRET_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # payment card numbers (with optional separators)
    re.compile(r"(?<!\d)(?:\d[ \-]?){13,19}(?!\d)"),
    # JWT
    re.compile(r"\beyJ[A-Za-z0-9_\-]{6,}\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]*"),
    # classic API keys / session tokens
    re.compile(r"\bsk-[A-Za-z0-9_\-]{6,}"),
    re.compile(r"\b(?:ghp|gho|ghs|ghr)_[A-Za-z0-9]{10,}"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{6,}"),
    re.compile(r"\bAKIA[0-9A-Z]{12,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{6,}"),
    # high-entropy single token (letters + digits, 24+ chars, no spaces)
    re.compile(
        r"(?<!\w)(?=[A-Za-z0-9+/=_\-]{24,}(?!\w))(?=[^\s]*[0-9])(?=[^\s]*[A-Za-z])"
        r"[A-Za-z0-9+/=_\-]{24,}(?!\w)"
    ),
)


def looks_secret(value: str) -> bool:
    """True when ``value`` looks like a credential rather than page text."""
    if not value:
        return False
    return any(p.search(value) for p in SECRET_VALUE_PATTERNS)


def redact_secrets(text: str) -> str:
    """Replace every secret-looking fragment of ``text`` with ``REDACTED``."""
    if not text:
        return text
    out = text
    for pattern in SECRET_VALUE_PATTERNS:
        out = pattern.sub(REDACTED, out)
    return out


def redact_args(
    args: dict[str, Any] | None,
    *,
    password_target: bool = False,
    sensitive_keys: Sequence[str] = ("text",),
) -> dict[str, Any]:
    """Copy ``args`` with credential-looking keys/values masked.

    ``password_target`` marks a call whose element is a password/credential
    field: in that case the *typed value* itself is dropped even if it does not
    look like a secret (``hunter2`` does not look like a card number).
    """

    def _clean(key: str, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: _clean(k, v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_clean(key, v) for v in value]
        if isinstance(value, str):
            if SECRET_KEY_RE.search(key):
                return REDACTED
            if password_target and key in sensitive_keys:
                return REDACTED
            return redact_secrets(value)
        return value

    return {str(k): _clean(str(k), v) for k, v in (args or {}).items()}


# --------------------------------------------------------------------------- #
# Signals and rules
# --------------------------------------------------------------------------- #

@dataclass
class Signals:
    """Everything a rule may look at, already normalised."""

    tool: str = ""
    kind: str = "other"
    label: str = ""
    placeholder: str = ""
    intent: str = ""
    role: str = ""
    field_type: str = ""
    key: str = ""
    action: str = ""
    href: str = ""
    href_host: str = ""
    url: str = ""
    host: str = ""
    path: str = ""
    title: str = ""
    flow: str = ""
    is_password_field: bool = False
    submit_requested: bool = False
    typed_text: str = ""

    def __post_init__(self) -> None:
        #: element-level wording only - never the page title (see module docs)
        self.text = " | ".join(p for p in (self.label, self.placeholder, self.intent) if p)

    def get(self, name: str) -> str:
        value = getattr(self, name, "")
        return value if isinstance(value, str) else ""


@dataclass(frozen=True)
class Rule:
    """One semantic rule."""

    id: str
    family: str
    level: RiskLevel
    reason: str
    pattern: re.Pattern[str] | None = None
    signals: tuple[str, ...] = ("label", "placeholder", "intent", "text")
    kinds: tuple[str, ...] = ()
    roles: tuple[str, ...] = ()
    negative: re.Pattern[str] | None = None
    guard: Callable[[Signals], bool] | None = None

    def applies(self, sig: Signals) -> bool:
        if self.kinds and sig.kind not in self.kinds:
            return False
        if self.roles and sig.role not in self.roles:
            return False
        if self.guard is not None and not self.guard(sig):
            return False
        return True

    def match(self, sig: Signals) -> tuple[str, str] | None:
        """Return ``(signal_name, matched_text)`` or ``None``."""
        if self.pattern is None:
            return ("guard", self.reason)
        for name in self.signals:
            value = sig.get(name)
            if not value:
                continue
            hit = self.pattern.search(value)
            if hit is None:
                continue
            if self.negative is not None and self.negative.search(value):
                continue
            return (name, hit.group(0).strip())
        return None


def _rx(pattern: str) -> re.Pattern[str]:
    """Compile a rule pattern: case-insensitive, unicode (Cyrillic-aware)."""
    return re.compile(pattern, re.IGNORECASE | re.UNICODE)


# -- wording tables (English + Russian) ------------------------------------- #

#: money leaves the account / an order is actually placed
_PAY_STRONG = (
    r"place\s+(?:my\s+|the\s+)?order"
    r"|complete\s+(?:my\s+|the\s+)?(?:order|purchase|payment|checkout|booking)"
    r"|confirm\s+(?:my\s+|the\s+)?(?:order|payment|purchase|booking|reservation|transfer)"
    r"|submit\s+(?:payment|order)"
    r"|pay\s+now|pay\s+with|make\s+a\s+(?:payment|donation)"
    r"|buy\s+now|place\s+(?:a\s+)?bid|book\s+now|reserve\s+now"
    r"|send\s+money|transfer\s+(?:money|funds)"
    r"|withdraw\s+(?:funds|cash|money)|cash\s+out"
    r"|\bdonate\b"
    # Russian
    r"|оплат(?:ить|ите|а|у|ы)"
    r"|оформ(?:ить|ляю|ление)\s+заказ"
    r"|подтверд(?:ить|ите)\s+заказ"
    r"|заказ(?:ать|ываю)"
    r"|куп(?:ить|лю|ите)"
    r"|приобрест(?:и|ите)"
    r"|перевест(?:и|ите)\s+(?:деньги|средства|сумму)"
    r"|перевод\s+(?:денег|средств)"
    r"|вывест(?:и|ите)\s+средства"
    r"|снят(?:ь|ите)\s+(?:деньги|средства)"
    r"|пожертв(?:овать|ование)"
    r"|заброниров(?:ать|ать)"
)

#: entering the payment flow (navigation) rather than committing to it
_PAY_FLOW_WORD = (
    r"\bcheck\s?-?out\b|\bcheckout\b|\bpayment\b|proceed\s+to\s+(?:payment|checkout|pay)"
    r"|\bоплатить\b|\bоплата\b|\bоформить\s+заказ\b|\bперейти\s+к\s+оплате\b"
)

_DELETE_STRONG = (
    r"\bdelete\b|\berase\b|\bdestroy\b|\bwipe\b|\bpurge\b|\bpermanently\s+delete\b"
    r"|\bdelete\s+(?:my\s+|the\s+)?account\b"
    r"|\bclose\s+(?:my\s+|the\s+)?account\b|\bdeactivate\s+(?:my\s+)?account\b"
    r"|\bempty\s+(?:the\s+)?(?:trash|bin|recycle\s+bin|folder)\b"
    r"|\bclear\s+all\b|\breset\s+all\b|\bdelete\s+all\b|\bremove\s+everything\b"
    r"|\bdiscard\s+(?:all\s+)?(?:changes|draft|edits)\b"
    r"|\bcancel\s+(?:my\s+|the\s+)?(?:order|subscription|booking|reservation|appointment|plan|account)\b"
    r"|\bunsubscribe\s+(?:from\s+)?all\b"
    # Russian
    r"|удал(?:ить|ите|и|яю|яет|ение|яется)"
    r"|стереть|стирать"
    r"|уничтож(?:ить|ение)"
    r"|очист(?:ить|ка)\s+(?:удал[её]нн\w*|мусор|вс[её])"
    r"|закрыть\s+(?:аккаунт|сч[её]т|профиль)"
    r"|деактивировать\s+аккаунт"
    r"|отменить\s+заказ"
    r"|отписаться\s+от\s+всех"
    r"|отменить\s+(?:подписку|бронирование)"
)

#: destructive verbs used by the mass-action rule
_MASS_VERB = (
    r"delete|remove|destroy|wipe|purge|send|email|invite|apply|unsubscribe|"
    r"cancel|archive|export|download"
    r"|удал|уничтож|отправ|приглас|отклик|отпис|очист|архив|экспорт|скача"
)

_MASS_QUANTIFIER = _rx(r"\b(?:all|every|everything|everyone|bulk|mass)\b|(?:все|всех|всё|целиком|оптом)")

_SEND_STRONG = (
    r"\bsend\b|\bsend\s+(?:message|email|mail|sms|feedback|invite|offer)\b"
    r"|\bsubmit\s+(?:application|form|review|request|claim|order|payment|ticket|response|answer|survey)\b"
    r"|\breply\b|\breply\s+all\b|\bforward\b"
    r"|\bpublish\b|\btweet\b|\bpost\s+(?:comment|reply|now)\b|\bcomment\b|\bshare\b"
    r"|\bapply\s+(?:now|for|to)\b|submit\s+(?:application|form|review)"
    r"|\bpost\b"
    # Russian
    r"|отправ(?:ить|ить\s+всем|ляю|ка|ить\s+сообщение)"
    r"|отправить|написать\s+(?:сообщение|отзыв|комментарий|письмо)"
    r"|ответить|переслать"
    r"|опубликовать|запостить|поделиться|прокомментировать"
    r"|оставить\s+(?:отзыв|комментарий|заявку)"
    r"|откликнуться|подать\s+заявку"
)

#: "Apply" the *filters*, not "Apply for the job"
_SEND_NEGATIVE = _rx(
    r"apply\s+(?:filter|filters|coupon|promo|discount|code|settings|changes|sort|search|tag|label)"
    r"|примен(?:ить|и)\s+(?:фильтр\w*|сортировк\w*|настройк\w*|промокод)"
    r"|отправить\s+форму\s+поиска"
)

_ACCOUNT_STRONG = (
    r"chang(?:e|ing)\s+(?:my\s+|the\s+)?password\b"
    r"|\breset\s+(?:my\s+|the\s+)?password\b|\bpassword\s+reset\b"
    r"|\bset\s+(?:a\s+)?new\s+password\b"
    r"|\bchang(?:e|ing)\s+(?:my\s+|the\s+)?(?:email|e-?mail|phone|number|username|login)\b"
    r"|\b(?:enable|disable|turn\s+(?:on|off))\s+(?:2fa|two[\s-]?factor|mfa)\b"
    r"|\b(?:two[\s-]?factor|2fa|mfa)\s+(?:authentication|settings|enrol\w*)\b"
    r"|\bgrant\s+(?:access|permission|admin|role)\b|\brevoke\s+(?:access|permission|token|role)\b"
    r"|\bremove\s+(?:user|member|teammate|guest)\b|\bmake\s+(?:me\s+|him\s+|her\s+)?admin\b"
    r"|\binvite\s+(?:user|member|people|teammate)\b"
    r"|\btransfer\s+ownership\b|\bchang(?:e|ing)\s+(?:my\s+)?plan\b|\bupgrade\s+(?:to|my\s+plan)\b"
    r"|\bdowngrade\b"
    # Russian
    r"|смен(?:ить|ите)\s+пароль|изменить\s+пароль|сбросить\s+пароль|новый\s+пароль"
    r"|смен(?:ить|ите)\s+(?:почт\w*|e-?mail|телефон|номер|логин)"
    r"|измени(?:ть|те)\s+(?:почт\w*|e-?mail|телефон|номер|логин|имя)"
    r"|двухфакторн\w*|\b2fa\b"
    r"|предоставить\s+доступ|отозвать\s+доступ|выдать\s+права|назначить\s+администратором"
    r"|пригласить\s+(?:пользовател\w*|участник\w*)|передать\s+права"
    r"|сменить\s+тариф|изменить\s+тариф|подключить\s+тариф"
)

_SESSION_CHANGE = _rx(
    r"\bsign\s?-?out\b|\blog\s?-?out\b|\bsign\s?-?off\b|\bend\s+session\b"
    r"|\bвыйти\b|\bвыход\b|\bзавершить\s+сеанс\b"
)

_CREDENTIAL_WORDS = _rx(
    r"\bpassword\b|\bpass\s?code\b|\bpassphrase\b|\bпароль\b|\bпин[\s-]?код\b"
    r"|\bpin\b|\bcvv\b|\bcvc\b|\bsecurity\s+code\b|\bcard\s+number\b|\bcredit\s+card\b"
    r"|\bномер\s+карты\b|\bкод\s+подтверждения\b|\bone[\s-]?time\s+(?:code|password)\b"
    r"|\botp\b|\bseed\s+phrase\b|\bprivate\s+key\b|\bapi[\s-]?key\b|\bsecret\b"
)

_UPLOAD_WORDS = _rx(
    r"\bupload\b|\battach\s+(?:a\s+)?file\b|\bchoose\s+file\b|\bselect\s+file\b"
    r"|\badd\s+attachment\b|\bзагрузить\s+файл\b|\bприкрепить\s+файл\b|\bвыбрать\s+файл\b"
    r"|\bзагрузка\s+файла\b"
)

_EXECUTABLE_RE = _rx(
    r"\.(?:exe|msi|msix|dmg|pkg|deb|rpm|appimage|apk|bat|cmd|scr|jar|vbs|ps1)"
    r"(?:\?|#|$|\s)"
)

_INSTALL_WORDS = _rx(
    r"\binstall\b|\binstaller\b|\bsetup\b|\brun\s+installer\b|\bdownload\s+and\s+run\b"
    r"|\bupdate\s+(?:the\s+)?app\b|\bустановить\b|\bустановка\b|\bзапустить\s+установщик\b"
)

_DOWNLOAD_WORDS = _rx(
    r"\bdownload\b|\bsave\s+file\b|\bsave\s+as\b|\bexport\b|\bскачать\b|\bсохранить\s+файл\b"
    r"|\bэкспорт\b"
)

_UNSUBSCRIBE_CAUTION = _rx(r"\bunsubscribe\b|\bотписаться\b|\bотписка\b")
_CANCEL_CAUTION = _rx(
    r"\bcancel\b|\bотмена\b|\bотменить\b|\bсбросить\b|\bотказаться\b"
)
_CART_SCOPE = _rx(
    r"(?:from\s+)?(?:cart|basket|bag|wishlist|favorites|favourites|compare|saved)"
    r"|корзин|избранн|сравнени|покупк|списк[ае]\s+желани"
)
_REMOVE_WORDS = _rx(
    r"\bremove\b|\bdelete\b|\bубрать\b|\bудалить\b|\bочистить\s+корзину\b|\bочист(?:ить|ка)\s+корзин\w*"
)

#: a plain "Submit"/"Отправить" outside a payment flow is a form commit, but it
#: is usually a search or settings form: caution, not destructive.
_SUBMIT_BARE = _rx(r"\bsubmit\b|\bотправить\s+форму\b|\bсохранить\s+и\s+отправить\b")

_FLOW_PAYMENT_PATH = _rx(
    r"checkout|/pay(?:/|$|\?)|payment|paynow|billing|payout|transfer|wallet"
    r"|place[-_]order|order/place|оформлени|оплат|плат(?:е|ё)ж"
)
_FLOW_PAYMENT_TITLE = _rx(
    r"checkout|payment|pay\s+now|billing|place\s+(?:your\s+)?order"
    r"|оплат|плат(?:е|ё)ж|оформлени\w*\s+заказ"
)
_FLOW_BOOKING = _rx(r"booking|reserve|reservation|appointment|ticket|забронир|запис(?:ь|и)\s+на")
_FLOW_CART = _rx(r"cart|basket|\bbag\b|корзин")

_PAYMENT_HOST = _rx(
    r"(?:^|\.)(?:paypal|stripe|wise|payoneer|revolut|qiwi|yoomoney|webmoney|skrill"
    r"|sberbank|tinkoff|alfabank|vtb|gazprombank|raiffeisen|rosbank|psbank|bank"
    r"|bankofamerica|chase|citi|wellsfargo|hsbc|barclays|monobank|privat24"
    r"|mastercard|visa|americanexpress|amex)\."
)


def _is_password_signal(sig: Signals) -> bool:
    return bool(sig.is_password_field) or sig.field_type == "password"


def _payment_flow(sig: Signals) -> bool:
    return sig.flow in ("payment", "booking")


def _not_payment_flow(sig: Signals) -> bool:
    return sig.flow not in ("payment", "booking")


def _looks_like_commit(sig: Signals) -> bool:
    """A button that commits a payment/booking form."""
    return sig.role in ("submit", "button") or sig.role not in ("link", "menuitem")


def default_rules() -> list[Rule]:
    """The rule table.  Order is irrelevant: the strictest match wins."""
    return [
        # ---------------- payments / checkout / order / money --------------- #
        Rule(
            id="pay.commit",
            family="payment",
            level="destructive",
            reason="places an order or moves money; hard to undo",
            pattern=_rx(_PAY_STRONG),
            signals=("label", "placeholder", "intent", "text"),
            kinds=("click", "type", "key"),
        ),
        Rule(
            id="pay.action",
            family="payment",
            level="destructive",
            reason="starts a payment; money can leave the account",
            pattern=_rx(r"\bpay\b|\bpurchase\b|\bcheckout\b|\bcheck\s?out\b|\bоплатить\b|\bоплата\b"),
            signals=("label", "intent", "text"),
            kinds=("click",),
            roles=("button", "submit", "menuitem", "tab", "option", "other"),
            guard=_looks_like_commit,
            negative=_rx(_PAY_STRONG),
        ),
        Rule(
            id="pay.enter_flow",
            family="payment",
            level="caution",
            reason="enters the payment/checkout flow (no money moved yet)",
            pattern=_rx(_PAY_FLOW_WORD),
            signals=("label", "placeholder", "intent", "text"),
            kinds=("click", "navigate"),
            guard=_not_payment_flow,
            negative=_rx(_PAY_STRONG),
        ),
        Rule(
            id="pay.payment_host",
            family="payment",
            level="destructive",
            reason="targets a bank/payment host; actions there move real money",
            pattern=_PAYMENT_HOST,
            signals=("href_host", "host"),
            kinds=("navigate", "click"),
        ),
        Rule(
            id="pay.checkout_path_commit",
            family="payment",
            level="destructive",
            reason="commits a form inside the payment/checkout flow",
            pattern=_rx(
                r"\b(submit|confirm|continue|next|place|pay|order|complete|finish|save|done|agree)\b"
                r"|подтверд\w*|продолж\w*|оформ\w*|заказ\w*|оплат\w*|отправ\w*|сохранить\w*"
            ),
            signals=("label", "intent", "text"),
            kinds=("click",),
            guard=_payment_flow,
            negative=_rx(
                r"continue\s+shopping|keep\s+shopping|go\s+back|\bcancel\b"
                r"|продолжить\s+покупки|вернуться\s+назад|отмена"
            ),
        ),
        # ---------------- deleting / destroying / mass actions --------------- #
        Rule(
            id="delete.destroy",
            family="delete",
            level="destructive",
            reason="deletes or destroys data or an account",
            pattern=_rx(_DELETE_STRONG),
            signals=("label", "placeholder", "intent", "text"),
            guard=lambda sig: not _CART_SCOPE.search(sig.text),
        ),
        Rule(
            id="delete.remove",
            family="delete",
            level="destructive",
            reason="removes content that is not easily restored",
            pattern=_rx(r"\bremove\b|\bубрать\b"),
            signals=("label", "placeholder", "intent", "text"),
            guard=lambda sig: not _CART_SCOPE.search(sig.text),
        ),
        Rule(
            id="delete.scoped",
            family="delete",
            level="caution",
            reason="removes something scoped to a cart/favourites list (recoverable)",
            pattern=_REMOVE_WORDS,
            signals=("label", "placeholder", "intent", "text"),
            guard=lambda sig: bool(_CART_SCOPE.search(sig.text)),
        ),
        Rule(
            id="delete.unsubscribe",
            family="delete",
            level="caution",
            reason="unsubscribes from a mailing list (account-level side effect)",
            pattern=_UNSUBSCRIBE_CAUTION,
            signals=("label", "placeholder", "intent", "text"),
            negative=_rx(r"\bunsubscribe\s+(?:from\s+)?all\b|отписаться\s+от\s+всех"),
        ),
        Rule(
            id="delete.cancel_bare",
            family="delete",
            level="caution",
            reason="cancels/aborts something (only harmful if it hits a checkout)",
            pattern=_CANCEL_CAUTION,
            signals=("label", "intent", "text"),
            negative=_rx(
                r"\b(?:order|subscription|booking|reservation|appointment|plan|account)\b"
                r"|заказ|подписк|бронир|аккаунт|сч[её]т"
            ),
        ),
        Rule(
            id="mass.unbounded",
            family="mass",
            level="destructive",
            reason="acts on all/every item at once; the scope is unbounded",
            pattern=_rx(_MASS_VERB),
            signals=("label", "placeholder", "intent", "text"),
            guard=lambda sig: bool(_MASS_QUANTIFIER.search(sig.text)),
        ),
        Rule(
            id="mass.select_all",
            family="mass",
            level="caution",
            reason="selects every item (usually the first half of a bulk action)",
            pattern=_rx(r"\bselect\s+all\b|\bвыбрать\s+все\b|\bвыделить\s+все\b|\bотметить\s+все\b"),
            signals=("label", "placeholder", "intent", "text"),
        ),
        # ---------------- sending / publishing / applying -------------------- #
        Rule(
            id="send.message",
            family="send",
            level="destructive",
            reason="sends/publishes something on the human's behalf",
            pattern=_rx(_SEND_STRONG),
            signals=("label", "placeholder", "intent", "text"),
            kinds=("click", "key"),
            negative=_SEND_NEGATIVE,
        ),
        Rule(
            id="send.type_and_submit_message",
            family="send",
            level="destructive",
            reason="types a message and submits it in one step (it will be sent)",
            pattern=_rx(
                r"\bmessage\b|\bcomment\b|\breply\b|\bnote\b|\bemail\b|\bchat\b"
                r"|сообщени|комментар|отзыв|письмо|ответ\b"
            ),
            signals=("label", "placeholder"),
            kinds=("type",),
            guard=lambda sig: sig.submit_requested,
        ),
        Rule(
            id="send.social",
            family="send",
            level="caution",
            reason="public or social state change (follow/like/subscribe)",
            pattern=_rx(
                r"\bfollow\b|\blike\b|\bunfollow\b|\bsubscribe\b|\badd\s+friend\b"
                r"|\bподписаться\b|\bлайк\b|\bотслеживать\b"
            ),
            signals=("label", "placeholder", "intent", "text"),
            kinds=("click",),
        ),
        Rule(
            id="send.submit_bare",
            family="send",
            level="caution",
            reason="commits a form (irreversible only if the form is transactional)",
            pattern=_SUBMIT_BARE,
            signals=("label", "placeholder", "intent", "text"),
            kinds=("click",),
        ),
        Rule(
            id="send.submit_shortcut",
            family="send",
            level="caution",
            reason="submits the focused form with the keyboard",
            pattern=None,
            kinds=("key",),
            guard=lambda sig: sig.key.lower() in ("enter", "return"),
        ),
        Rule(
            id="send.type_and_submit",
            family="send",
            level="caution",
            reason="types a value and submits the form in one step",
            pattern=None,
            kinds=("type",),
            guard=lambda sig: sig.submit_requested,
        ),
        # ---------------- account / permissions / credentials ---------------- #
        Rule(
            id="account.change",
            family="account",
            level="destructive",
            reason="changes account security, identity or permissions",
            pattern=_rx(_ACCOUNT_STRONG),
            signals=("label", "placeholder", "intent", "text"),
        ),
        Rule(
            id="account.session",
            family="account",
            level="caution",
            reason="ends or starts a session (not destructive)",
            pattern=_SESSION_CHANGE,
            signals=("label", "placeholder", "intent", "text"),
        ),
        Rule(
            id="account.forgot_password",
            family="account",
            level="caution",
            reason="starts a credential-reset flow (triggers an email/notification)",
            pattern=_rx(r"\bforgot\s+password\b|\bforgot\s+your\s+password\b|\bзабыли\s+пароль\b"),
            signals=("label", "placeholder", "intent", "text"),
        ),
        Rule(
            id="credential.password_field",
            family="credential",
            level="destructive",
            reason="would type a secret into a password/credential field",
            pattern=None,
            kinds=("type",),
            guard=_is_password_signal,
        ),
        Rule(
            id="credential.named_field",
            family="credential",
            level="destructive",
            reason="the field expects a credential (password, PIN, card, OTP)",
            pattern=_CREDENTIAL_WORDS,
            signals=("label", "placeholder"),
            kinds=("type",),
        ),
        Rule(
            id="credential.card_payment",
            family="credential",
            level="destructive",
            reason="enters payment-card data",
            pattern=_rx(r"\bcard\s+number\b|\bномер\s+карты\b|\bcvv\b|\bcvc\b|\bsecurity\s+code\b"),
            signals=("label", "placeholder", "intent", "text"),
        ),
        # ---------------- file upload -------------------------------------- #
        Rule(
            id="upload.file",
            family="upload",
            level="destructive",
            reason="uploads a local file to a remote site (data leaves the machine)",
            pattern=_UPLOAD_WORDS,
            signals=("label", "placeholder", "intent", "text"),
            kinds=("click", "type"),
        ),
        Rule(
            id="upload.file_field",
            family="upload",
            level="destructive",
            reason="this element is a file-upload control",
            pattern=None,
            kinds=("click", "type"),
            guard=lambda sig: sig.field_type == "file" or sig.role == "file",
        ),
        # ---------------- downloads / installs ------------------------------ #
        Rule(
            id="download.executable",
            family="download",
            level="destructive",
            reason="downloads or installs an executable",
            pattern=_EXECUTABLE_RE,
            signals=("href", "url", "label", "intent", "text"),
        ),
        Rule(
            id="download.install",
            family="download",
            level="destructive",
            reason="installs software on the human's machine",
            pattern=_INSTALL_WORDS,
            signals=("label", "intent", "text"),
        ),
        Rule(
            id="download.generic",
            family="download",
            level="caution",
            reason="downloads or exports a file",
            pattern=_DOWNLOAD_WORDS,
            signals=("label", "intent", "text"),
        ),
        # ---------------- browser-level state ------------------------------ #
        Rule(
            id="tabs.close",
            family="browser",
            level="caution",
            reason="closes a browser tab (the run may lose its page)",
            pattern=_rx(r"\bclose\b|\bзакрыть\b"),
            signals=("action", "intent", "label"),
            kinds=("tabs",),
        ),
    ]


#: The rule table used by :class:`SecurityPolicy` unless one is injected.
RULES: list[Rule] = default_rules()


# --------------------------------------------------------------------------- #
# The policy
# --------------------------------------------------------------------------- #

_KIND_BY_TOOL: dict[str, str] = {
    "goto": "navigate",
    "go_back": "navigate",
    "click": "click",
    "type_text": "type",
    "press_key": "key",
    "scroll": "scroll",
    "tabs": "tabs",
    "wait_for": "wait",
}


def _host_of(url: str) -> str:
    """Host part of a URL (lower-cased, port stripped); ``''`` for relative URLs."""
    if not url:
        return ""
    match = re.match(r"^\s*[a-zA-Z][a-zA-Z0-9+.\-]*://([^/?#]+)", url)
    if match:
        return match.group(1).split("@")[-1].split(":")[0].lower()
    if url.startswith("/") or url.startswith("?"):
        return ""
    return url.split("/")[0].split("?")[0].split(":")[0].lower()


def _path_of(url: str) -> str:
    if not url:
        return ""
    match = re.match(r"^\s*[a-zA-Z][a-zA-Z0-9+.\-]*://[^/?#]*([^?#]*)", url)
    if match:
        return match.group(1) or "/"
    return url.split("?")[0] or url


def host_matches(host: str, domains: Iterable[str]) -> bool:
    """Suffix match on the page host, safe against ``example.com.evil.net``."""
    host = (host or "").lower().strip(".")
    if not host:
        return False
    for raw in domains or ():
        domain = (raw or "").lower().strip().strip(".")
        if not domain:
            continue
        if host == domain or host.endswith("." + domain):
            return True
    return False


class SecurityPolicy:
    """Classify tool calls by meaning and decide who may run them."""

    def __init__(
        self,
        config: Config,
        llm: Any = None,
        *,
        rules: Sequence[Rule] | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.rules: list[Rule] = list(rules) if rules is not None else list(RULES)
        self._lock = threading.RLock()
        self.audit_errors: list[str] = []

    # ------------------------------------------------------------------ #
    # classification
    # ------------------------------------------------------------------ #
    def signals(self, call: ToolCall, ctx: ActionContext) -> Signals:
        args = dict(call.args or {})
        element: Element | None = ctx.element
        url = str(args.get("url") or ctx.url or "")
        href = ""
        if element is not None and element.href:
            href = element.href
        elif isinstance(args.get("url"), str) and call.name == "goto":
            href = str(args["url"])
        intent = str(args.get("intent") or "")
        if element is not None:
            label = element.name or intent
            placeholder = element.placeholder or ""
            field_type = element.type or ""
            role = element.role or ""
            if not href and element.href:
                href = element.href
        else:
            label = intent
            placeholder = ""
            field_type = ""
            role = ""
            if call.name == "click":
                role = "button"
            elif call.name == "type_text":
                role = "textbox"
        host = _host_of(ctx.url) if ctx.url else _host_of(url)
        sig = Signals(
            tool=call.name,
            kind=_KIND_BY_TOOL.get(call.name, "other"),
            label=label,
            placeholder=placeholder,
            intent=intent,
            role=role,
            field_type=field_type,
            key=str(args.get("key") or ""),
            action=str(args.get("action") or ""),
            href=href,
            href_host=_host_of(href),
            url=url,
            host=host or _host_of(href),
            path=_path_of(ctx.url or url),
            title=ctx.page_title or "",
            is_password_field=bool(ctx.is_password_field) or field_type == "password",
            submit_requested=bool(args.get("submit")),
            typed_text=str(args.get("text") or ""),
        )
        sig.flow = self._flow(sig)
        return sig

    @staticmethod
    def _flow(sig: Signals) -> str:
        """Is this page a checkout/payment, booking or cart page?"""
        path, title = sig.path or "", sig.title or ""
        if _FLOW_PAYMENT_PATH.search(path) or _FLOW_PAYMENT_TITLE.search(title):
            return "payment"
        if _FLOW_BOOKING.search(path) or _FLOW_BOOKING.search(title):
            return "booking"
        if _FLOW_CART.search(path) or _FLOW_CART.search(title):
            return "cart"
        return ""

    def classify(self, call: ToolCall, ctx: ActionContext) -> ActionRisk:
        """Return the risk of ``call`` in ``ctx`` (deterministic + optional LLM)."""
        sig = self.signals(call, ctx)
        risk = self._classify_signals(call, ctx, sig)
        return risk

    def _classify_signals(self, call: ToolCall, ctx: ActionContext, sig: Signals) -> ActionRisk:
        matched: list[Rule] = []
        reasons: list[str] = []
        for rule in self.rules:
            if not rule.applies(sig):
                continue
            hit = rule.match(sig)
            if hit is None:
                continue
            matched.append(rule)
            signal_name, evidence = hit
            if rule.pattern is None:
                detail = f"[{rule.id}]"
            else:
                detail = f"[{rule.id}: matched {evidence!r} in {signal_name}]"
            reasons.append(f"{rule.reason} {detail}")

        level: RiskLevel = "safe"
        if any(r.level == "caution" for r in matched):
            level = "caution"
        if any(r.level == "destructive" for r in matched):
            level = "destructive"

        risk = ActionRisk(
            level=level,
            reasons=reasons[:6],
            matched_rules=[r.id for r in matched],
        )
        if (
            risk.level == "caution"
            and getattr(self.config, "llm_risk_check", False)
            and self.llm is not None
        ):
            risk = self._second_opinion(call, ctx, sig, risk)
        return risk

    # ------------------------------------------------------------------ #
    # optional LLM second opinion (caution only, never required)
    # ------------------------------------------------------------------ #
    def _second_opinion(
        self, call: ToolCall, ctx: ActionContext, sig: Signals, risk: ActionRisk
    ) -> ActionRisk:
        prompt = (
            "A browser-automation agent wants to perform ONE action. Judge only "
            "this action, not the site.\n"
            f"tool: {call.name}\n"
            f"arguments: {json.dumps(redact_args(call.args, password_target=sig.is_password_field), ensure_ascii=False)}\n"
            f"element: role={sig.role or '?'} name={sig.label!r} type={sig.field_type or '?'}\n"
            f"page: {ctx.url or '?'} - {ctx.page_title or '?'}\n"
            f"deterministic signals: {'; '.join(risk.reasons) or 'none'}\n\n"
            "Could this single action move money, delete or destroy data, send or "
            "publish something as the user, change account security/permissions, "
            "hand over credentials, or upload a local file?\n"
            'Answer with strict JSON only: {"level": "safe|caution|destructive", '
            '"reason": "<max 20 words>"}'
        )
        try:
            response = self.llm.complete(
                [LLMMessage(role="user", content=prompt)],
                tools=None,
                system=(
                    "You are a strict security reviewer for an autonomous browser "
                    "agent. Reply with JSON and nothing else."
                ),
                temperature=0.0,
                max_tokens=120,
            )
            verdict = self._parse_verdict(getattr(response, "text", "") or "")
        except Exception as exc:  # noqa: BLE001 - any provider/transport failure
            risk.reasons = risk.reasons + [f"[llm.degraded: second opinion unavailable ({type(exc).__name__})]"]
            return risk
        if verdict is None:
            risk.reasons = risk.reasons + ["[llm.degraded: unparsable second opinion, keeping rule verdict]"]
            return risk
        level, reason = verdict
        if level == risk.level:
            risk.reasons = risk.reasons + [f"[llm.agree: {reason}]"]
            return risk
        if level == "destructive":
            risk.level = "destructive"
            risk.matched_rules = list(risk.matched_rules) + ["llm.escalated"]
            risk.reasons = risk.reasons + [f"[llm.escalated: {reason}]"]
            return risk
        if level == "safe":
            risk.level = "safe"
            risk.matched_rules = list(risk.matched_rules) + ["llm.downgraded"]
            risk.reasons = risk.reasons + [f"[llm.downgraded: {reason}]"]
            return risk
        risk.reasons = risk.reasons + [f"[llm.note: {reason}]"]
        return risk

    @staticmethod
    def _parse_verdict(text: str) -> tuple[RiskLevel, str] | None:
        if not text:
            return None
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", cleaned).strip()
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except (ValueError, TypeError):
            return None
        if not isinstance(data, dict):
            return None
        level = str(data.get("level", "")).strip().lower()
        if level not in ("safe", "caution", "destructive"):
            return None
        return level, str(data.get("reason", ""))[:200]  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    # authorization
    # ------------------------------------------------------------------ #
    def _auto_approved_host(self, sig: Signals) -> str:
        """Host (current page or navigation target) that is allow-listed, or ''."""
        domains = list(getattr(self.config, "auto_approve_domains", None) or [])
        if not domains:
            return ""
        for host in (sig.host, _host_of(sig.url), sig.href_host):
            if host and host_matches(host, domains):
                return host
        return ""

    def confirmation_text(
        self, call: ToolCall, risk: ActionRisk, ctx: ActionContext
    ) -> tuple[str, str]:
        """``(prompt, details)`` for the human, credential-safe."""
        sig = self.signals(call, ctx)
        if call.name == "click" and sig.label:
            action = f"click '{sig.label}'"
        elif call.name == "type_text":
            where = sig.label or sig.placeholder or f"field #{call.args.get('element_id')}"
            action = f"type into '{where}'"
        elif call.name == "goto":
            action = f"navigate to {sig.url}"
        else:
            action = call.name
        prompt = f"Allow: {action}?"
        element = ctx.element.line() if ctx.element is not None else "(no element)"
        details_lines = [
            f"tool: {call.name}",
            "arguments: "
            + json.dumps(redact_args(call.args, password_target=sig.is_password_field), ensure_ascii=False),
            f"element: {element}",
            f"page: {ctx.url or '?'} - {ctx.page_title or '?'}",
            f"risk: {risk.level}",
            "why:",
        ]
        details_lines += [f"  - {r}" for r in (risk.reasons or ["no rule details"])]
        if risk.matched_rules:
            details_lines.append("rules: " + ", ".join(risk.matched_rules))
        details_lines.append(
            "This action is treated as destructive: it is hard to undo, moves money "
            "or publishes something as you."
        )
        return prompt, "\n".join(details_lines)

    def authorize(
        self,
        call: ToolCall,
        risk: ActionRisk,
        ctx: ActionContext,
        confirm: ConfirmFn,
    ) -> tuple[bool, str]:
        """Decide whether ``call`` may run.  Always writes one audit line."""
        sig = self.signals(call, ctx)
        approved = True
        reason = f"{risk.level} action allowed without confirmation"
        human: bool | None = None
        decision = "allowed"

        if risk.level != "destructive":
            decision = "allowed_no_confirmation"
        elif not getattr(self.config, "allow_destructive_tools", True):
            approved = False
            decision = "refused"
            reason = (
                "destructive tool use is disabled (allow_destructive_tools=False); "
                "webpilot will not run this action at all"
            )
        elif self.config.confirm_mode == "deny":
            approved = False
            decision = "denied_by_config"
            reason = "confirm_mode=deny: every destructive action is refused"
        elif (
            sig.is_password_field
            and not getattr(self.config, "allow_password_typing", False)
            and "credential" in [r.split(".")[0] for r in risk.matched_rules]
        ):
            approved = False
            decision = "refused"
            reason = (
                "password fields are filled by the human (allow_password_typing=False); "
                "ask the human to log in"
            )
        elif self.config.confirm_mode == "allow":
            decision = "auto_approved"
            reason = "confirm_mode=allow: destructive actions are approved automatically"
        elif self._auto_approved_host(sig):
            decision = "auto_approved_domain"
            reason = f"host {self._auto_approved_host(sig)!r} is in auto_approve_domains"
        else:
            prompt, details = self.confirmation_text(call, risk, ctx)
            prompt_failed = False
            try:
                human = bool(confirm(prompt, details))
            except Exception as exc:  # noqa: BLE001 - a broken UI must not approve
                human = False
                prompt_failed = True
                reason = f"confirmation prompt failed ({type(exc).__name__}): treated as denied"
            if human:
                approved = True
                decision = "approved_by_human"
                reason = "the human approved this destructive action"
            else:
                approved = False
                decision = "denied_by_human"
                if not prompt_failed:
                    reason = "the human denied this destructive action"

        self.audit(call, risk, ctx, approved=approved, decision=decision, reason=reason, human=human)
        return approved, reason

    # ------------------------------------------------------------------ #
    # audit trail
    # ------------------------------------------------------------------ #
    def audit(
        self,
        call: ToolCall,
        risk: ActionRisk,
        ctx: ActionContext,
        *,
        approved: bool,
        decision: str,
        reason: str,
        human: bool | None = None,
    ) -> dict[str, Any]:
        """Append one JSON line to ``config.audit_path``; return the record."""
        sig = self.signals(call, ctx)
        record: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "tool": call.name,
            "args": redact_args(call.args, password_target=sig.is_password_field),
            "url": ctx.url,
            "title": ctx.page_title,
            "element": ctx.element.line() if ctx.element is not None else None,
            "risk": risk.level,
            "rules": list(risk.matched_rules),
            "reasons": list(risk.reasons),
            "decision": decision,
            "approved": bool(approved),
            "human_answer": human,
            "confirm_mode": self.config.confirm_mode,
            "reason": reason,
        }
        path = Path(self.config.audit_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(record, ensure_ascii=False)
            with self._lock:
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except OSError as exc:  # never crash the run over the audit file
            self.audit_errors.append(f"{type(exc).__name__}: {exc}")
        return record

    def audit_records(self) -> list[dict[str, Any]]:
        """Read the audit trail back (used by tests and by `webpilot audit`)."""
        path = Path(self.config.audit_path)
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
        return records

    # ------------------------------------------------------------------ #
    # presentation helper used by the CLI
    # ------------------------------------------------------------------ #
    def describe_risk(self, risk: ActionRisk) -> str:
        if risk.level == "safe" and not risk.matched_rules:
            return "safe"
        return f"{risk.level} ({', '.join(risk.matched_rules)})"
