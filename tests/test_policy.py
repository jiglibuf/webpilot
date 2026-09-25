"""Tests for :mod:`webpilot.security.policy`.

Everything here is offline: a ``Config(provider="fake")`` in a ``tmp_path``, no
browser, no API key, no network.  The suite pins three things:

* each risk family is detected from *meaning* in **both** English and Russian;
* the false-positive list stays safe (Search / Add to cart / View order history /
  Sign in / Sign out ...);
* the gate itself - confirm modes, domain allow-list, password refusal, LLM
  degradation and the redacting audit trail.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from webpilot.config import Config
from webpilot.errors import LLMError
from webpilot.security.policy import (
    REDACTED,
    RULES,
    SecurityPolicy,
    default_rules,
    host_matches,
    looks_secret,
    redact_args,
    redact_secrets,
)
from webpilot.types import (
    ActionContext,
    ActionRisk,
    Element,
    LLMMessage,
    LLMResponse,
    ToolCall,
    Usage,
)

PASSWORD = "hunter2-CORRECT-horse-battery"      # 27 chars, no digit-letter token
CARD = "4111111111111111"
API_KEY = "sk-fake-0123456789abcdefghij"   # synthetic, matches the secret pattern on purpose
JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abcdEFGH1234"

SITE = "https://shop.example.invalid"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

@pytest.fixture
def make_policy(tmp_path: Path):
    def _make(*, llm=None, rules=None, **config_kwargs):
        kwargs = dict(
            provider="fake",
            model="fake-1",
            headless=True,
            user_data_dir=tmp_path / "profile",
            transcript_dir=tmp_path / "transcripts",
            screenshot_dir=tmp_path / "screenshots",
            audit_path=tmp_path / "audit.jsonl",
        )
        kwargs.update(config_kwargs)
        config = Config(**kwargs)
        config.ensure_dirs()
        return SecurityPolicy(config, llm=llm, rules=rules)

    return _make


@pytest.fixture
def policy(make_policy):
    return make_policy()


def element(role: str = "button", name: str = "", **kw) -> Element:
    kw.setdefault("id", 7)
    kw.setdefault("tag", "button")
    return Element(role=role, name=name, **kw)  # type: ignore[arg-type]


def click(label: str, *, role: str = "button", url: str = f"{SITE}/page",
          title: str = "Page", intent: str = "", element_kw: dict | None = None,
          password: bool = False) -> tuple[ToolCall, ActionContext]:
    el = element(role, label, **(element_kw or {}))
    call = ToolCall("click", {"element_id": el.id, "intent": intent})
    return call, ActionContext(url=url, page_title=title, element=el,
                               is_password_field=password)


def typing(label: str, text: str, *, field_type: str = "text", url: str = f"{SITE}/page",
           title: str = "Page", role: str = "textbox", submit: bool = False,
           password: bool = False) -> tuple[ToolCall, ActionContext]:
    el = element(role, label, type=field_type)  # type: ignore[arg-type]
    call = ToolCall("type_text", {"element_id": el.id, "text": text, "submit": submit})
    return call, ActionContext(url=url, page_title=title, element=el,
                              is_password_field=password)


def level(policy: SecurityPolicy, call: ToolCall, ctx: ActionContext) -> str:
    return policy.classify(call, ctx).level


def destructive(policy: SecurityPolicy, call: ToolCall, ctx: ActionContext) -> ActionRisk:
    risk = policy.classify(call, ctx)
    assert risk.level == "destructive", f"expected destructive, got {risk}"
    assert risk.requires_confirmation is True
    assert risk.reasons, "a destructive verdict must explain itself"
    assert risk.matched_rules, "a destructive verdict must name its rules"
    return risk


def not_destructive(policy: SecurityPolicy, call: ToolCall, ctx: ActionContext) -> ActionRisk:
    risk = policy.classify(call, ctx)
    assert risk.level != "destructive", f"false positive: {risk}"
    assert risk.requires_confirmation is False
    return risk


class ScriptedLLM:
    """Minimal LLMClient double: records calls, returns canned text or raises."""

    provider = "fake"
    model = "fake-risk"

    def __init__(self, text: str = "", error: Exception | None = None):
        self.text = text
        self.error = error
        self.calls: list[dict] = []

    def complete(self, messages, tools=None, *, system=None, temperature=0.0, max_tokens=2048):
        self.calls.append({"messages": list(messages), "system": system,
                           "temperature": temperature, "max_tokens": max_tokens})
        if self.error is not None:
            raise self.error
        return LLMResponse(text=self.text, usage=Usage(11, 3, 0, 1))

    def count_tokens(self, text: str) -> int:  # pragma: no cover - unused here
        return len(text) // 4


# --------------------------------------------------------------------------- #
# 1. payments / checkout / orders / money transfer
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "label,kw",
    [
        ("Place order", {"url": f"{SITE}/checkout", "title": "Checkout"}),
        ("Complete purchase", {}),
        ("Confirm payment", {}),
        ("Pay now", {}),
        ("Buy now", {}),
        ("Send money", {}),
        ("Transfer funds", {}),
        ("Book now", {"url": f"{SITE}/booking", "title": "Booking"}),
        ("Checkout", {"role": "submit", "element_kw": {"type": "submit"}}),
    ],
)
def test_payment_wording_is_destructive_in_english(policy, label, kw):
    call, ctx = click(label, **kw)
    risk = destructive(policy, call, ctx)
    assert any(r.startswith("pay.") for r in risk.matched_rules)


@pytest.mark.parametrize(
    "label",
    [
        "Оплатить",
        "Оплатить заказ",
        "Оформить заказ",
        "Заказать",
        "Подтвердить заказ",
        "Купить сейчас",
        "Перевести средства",
        "Пожертвовать",
    ],
)
def test_payment_wording_is_destructive_in_russian(policy, label):
    call, ctx = click(label, url=f"{SITE}/checkout", title="Оформление заказа")
    risk = destructive(policy, call, ctx)
    assert any(r.startswith("pay.") for r in risk.matched_rules)


def test_navigation_to_a_bank_or_payment_host_is_destructive(policy):
    for url in ("https://www.sberbank.ru/private", "https://paypal.com/signin",
                "https://checkout.stripe.com/session", "http://alfabank.ru/x"):
        call = ToolCall("goto", {"url": url})
        risk = destructive(policy, call, ActionContext(url=url))
        assert "pay.payment_host" in risk.matched_rules, url


def test_plain_navigation_is_not_destructive(policy):
    for url in (f"{SITE}/search?q=chairs", "https://news.ycombinator.com/", f"{SITE}/cart"):
        call = ToolCall("goto", {"url": url})
        not_destructive(policy, call, ActionContext(url=url))


def test_entering_a_checkout_flow_by_link_is_only_a_caution(policy):
    """A link to /checkout is navigation; the commit happens on the next page."""
    call, ctx = click("Checkout", role="link", url=f"{SITE}/cart", title="Your cart")
    risk = policy.classify(call, ctx)
    assert risk.level == "caution"
    assert "pay.enter_flow" in risk.matched_rules


# --------------------------------------------------------------------------- #
# 2. deleting / destroying data
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "label,kw",
    [
        ("Delete account", {}),
        ("Delete", {}),
        ("Empty trash", {}),
        ("Permanently delete", {}),
        ("Remove", {}),
        ("Unsubscribe from all", {}),
        ("Cancel order", {}),
        ("Cancel my subscription", {}),
        ("Close my account", {}),
        ("Discard changes", {}),
        ("Clear all", {}),
    ],
)
def test_delete_wording_is_destructive_in_english(policy, label, kw):
    call, ctx = click(label, **kw)
    risk = destructive(policy, call, ctx)
    assert any(r.startswith(("delete.", "mass.")) for r in risk.matched_rules)


@pytest.mark.parametrize(
    "label",
    ["Удалить", "Удалить аккаунт", "Удалить все", "Стереть данные",
     "Отписаться от всех", "Отменить заказ", "Закрыть аккаунт", "Очистить удалённые"],
)
def test_delete_wording_is_destructive_in_russian(policy, label):
    call, ctx = click(label, url=f"{SITE}/settings", title="Настройки")
    risk = destructive(policy, call, ctx)
    assert any(r.startswith(("delete.", "mass.")) for r in risk.matched_rules)


def test_removing_from_a_cart_is_only_a_caution(policy):
    """Removing a cart line is recoverable - flag it, do not stop the demo."""
    call, ctx = click("Remove from cart", url=f"{SITE}/cart", title="Your cart")
    risk = policy.classify(call, ctx)
    assert risk.level == "caution"
    assert "delete.scoped" in risk.matched_rules


def test_cancel_alone_is_only_a_caution(policy):
    call, ctx = click("Cancel", title="Confirm")
    not_destructive(policy, call, ctx)


def test_unsubscribe_from_a_single_list_is_only_a_caution(policy):
    call, ctx = click("Unsubscribe")
    risk = not_destructive(policy, call, ctx)
    assert risk.level == "caution"


# --------------------------------------------------------------------------- #
# 3. sending / publishing / applying
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "label",
    ["Send", "Send message", "Reply", "Reply all", "Forward", "Publish",
     "Tweet", "Post", "Share", "Submit application", "Apply now", "Apply for this job"],
)
def test_send_and_publish_wording_is_destructive_in_english(policy, label):
    call, ctx = click(label)
    risk = destructive(policy, call, ctx)
    assert "send.message" in risk.matched_rules


@pytest.mark.parametrize(
    "label",
    ["Отправить", "Отправить сообщение", "Ответить", "Переслать", "Опубликовать",
     "Поделиться", "Оставить отзыв", "Откликнуться", "Подать заявку", "Написать сообщение"],
)
def test_send_and_publish_wording_is_destructive_in_russian(policy, label):
    call, ctx = click(label)
    risk = destructive(policy, call, ctx)
    assert "send.message" in risk.matched_rules


def test_apply_filters_is_not_destructive(policy):
    assert policy.classify(*click("Apply filters", url=f"{SITE}/search",
                                  title="Search")).level == "safe"
    assert policy.classify(*click("Применить фильтр")).level == "safe"


def test_pressing_enter_is_a_caution(policy):
    risk = policy.classify(ToolCall("press_key", {"key": "Enter"}), ActionContext(url=SITE))
    assert risk.level == "caution"
    assert "send.submit_shortcut" in risk.matched_rules


# --------------------------------------------------------------------------- #
# 4. account / permissions
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "label",
    ["Change password", "Reset password", "Enable 2FA", "Disable two-factor",
     "Change email", "Change my phone number", "Grant access", "Revoke access",
     "Invite user", "Make me admin", "Transfer ownership", "Change plan", "Downgrade"],
)
def test_account_and_permission_wording_is_destructive_in_english(policy, label):
    call, ctx = click(label, url=f"{SITE}/settings", title="Account settings")
    risk = destructive(policy, call, ctx)
    assert "account.change" in risk.matched_rules


@pytest.mark.parametrize(
    "label",
    ["Сменить пароль", "Изменить пароль", "Сбросить пароль", "Включить 2FA",
     "Двухфакторная аутентификация", "Изменить почту", "Сменить телефон",
     "Предоставить доступ", "Отозвать доступ", "Выдать права",
     "Назначить администратором", "Пригласить пользователя"],
)
def test_account_and_permission_wording_is_destructive_in_russian(policy, label):
    call, ctx = click(label, url=f"{SITE}/settings", title="Настройки")
    risk = destructive(policy, call, ctx)
    assert "account.change" in risk.matched_rules


def test_session_wording_is_never_destructive(policy):
    for label in ("Sign out", "Log out", "Sign in", "Log in", "Войти", "Выйти", "Выход"):
        risk = not_destructive(policy, *click(label, url=f"{SITE}/login", title="Sign in"))
        assert risk.level in ("safe", "caution"), label


def test_forgot_password_is_only_a_caution(policy):
    call, ctx = click("Forgot password?", url=f"{SITE}/login", title="Sign in")
    risk = not_destructive(policy, call, ctx)
    assert risk.level == "caution"


# --------------------------------------------------------------------------- #
# 5. credential entry
# --------------------------------------------------------------------------- #

def test_typing_into_a_password_field_is_destructive(policy):
    call, ctx = typing("Password", PASSWORD, field_type="password",
                       url=f"{SITE}/login", title="Sign in", password=True)
    risk = destructive(policy, call, ctx)
    assert "credential.password_field" in risk.matched_rules
    assert any("password" in r.lower() or "credential" in r.lower() for r in risk.reasons)


def test_russian_password_label_is_destructive(policy):
    call, ctx = typing("Пароль", PASSWORD, field_type="password",
                       url=f"{SITE}/login", title="Вход", password=True)
    destructive(policy, call, ctx)


@pytest.mark.parametrize(
    "label,field_type",
    [("Card number", "text"), ("CVV", "text"), ("PIN", "password"),
     ("One-time code", "text"), ("Номер карты", "text"), ("Код подтверждения", "text"),
     ("API key", "text")],
)
def test_credential_fields_are_destructive_to_type_into(policy, label, field_type):
    call, ctx = typing(label, "x" * 16, field_type=field_type)
    risk = destructive(policy, call, ctx)
    assert any(r.startswith("credential.") for r in risk.matched_rules)


def test_typing_ordinary_text_is_safe(policy):
    call, ctx = typing("Search products", "office chair", role="textbox")
    assert level(policy, call, ctx) == "safe"
    # typed content is *not* a signal: mentioning "delete" in a comment is not
    # the same as clicking a delete button
    call, ctx = typing("Comment", "please delete my account", role="textbox")
    assert level(policy, call, ctx) == "safe"


def test_typing_and_submitting_a_message_is_destructive(policy):
    call, ctx = typing("Message", "hello there", role="textbox", submit=True)
    destructive(policy, call, ctx)
    call, ctx = typing("Comment", "nice post", role="textbox", submit=True)
    destructive(policy, call, ctx)
    call, ctx = typing("Сообщение", "привет", role="textbox", submit=True)
    destructive(policy, call, ctx)


def test_typing_and_submitting_a_search_form_is_only_a_caution(policy):
    call, ctx = typing("Search products", "chair", role="textbox", submit=True)
    risk = policy.classify(call, ctx)
    assert risk.level == "caution"
    assert "send.type_and_submit" in risk.matched_rules


# --------------------------------------------------------------------------- #
# 6. uploads, 7. irreversible submits, 8. downloads/installs
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("label,kw", [
    ("Upload file", {}),
    ("Загрузить файл", {}),
    ("Attach file", {}),
    ("Choose file", {"role": "file", "element_kw": {"type": "file"}}),
])
def test_file_upload_is_destructive(policy, label, kw):
    call, ctx = click(label, **kw)
    risk = destructive(policy, call, ctx)
    assert any(r.startswith("upload.") for r in risk.matched_rules)


def test_file_input_is_destructive_even_with_an_innocent_label(policy):
    call, ctx = click("Browse", role="file", element_kw={"type": "file"})
    risk = destructive(policy, call, ctx)
    assert "upload.file_field" in risk.matched_rules


def test_irreversible_submit_inside_a_checkout_flow_is_destructive(policy):
    call, ctx = click("Continue", url=f"{SITE}/checkout/payment", title="Payment")
    risk = destructive(policy, call, ctx)
    assert "pay.checkout_path_commit" in risk.matched_rules


def test_the_same_submit_outside_a_checkout_flow_is_only_a_caution(policy):
    call, ctx = click("Submit", url=f"{SITE}/search", title="Search")
    risk = not_destructive(policy, call, ctx)
    assert risk.level == "caution"
    assert "send.submit_bare" in risk.matched_rules


@pytest.mark.parametrize("label", ["Install", "Run installer", "Установить", "Setup"])
def test_installs_are_destructive(policy, label):
    call, ctx = click(label)
    risk = destructive(policy, call, ctx)
    assert "download.install" in risk.matched_rules


def test_downloading_an_executable_is_destructive(policy):
    el = element("link", "Download", tag="a", href="https://cdn.example.invalid/wp-setup.exe")
    call = ToolCall("click", {"element_id": el.id, "intent": "download the tool"})
    ctx = ActionContext(url=SITE, page_title="Downloads", element=el)
    risk = destructive(policy, call, ctx)
    assert "download.executable" in risk.matched_rules


def test_downloading_an_ordinary_file_is_only_a_caution(policy):
    el = element("link", "Download invoice", tag="a", href="https://cdn.example.invalid/inv.pdf")
    call = ToolCall("click", {"element_id": el.id, "intent": "download the invoice"})
    ctx = ActionContext(url=SITE, page_title="Invoice", element=el)
    risk = not_destructive(policy, call, ctx)
    assert risk.level == "caution"


# --------------------------------------------------------------------------- #
# 9. mass / unbounded actions
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("label", ["Delete all", "Remove all", "Отправить всем",
                                   "Удалить все", "Select all and delete",
                                   "Unsubscribe from all newsletters"])
def test_mass_actions_are_destructive(policy, label):
    call, ctx = click(label)
    risk = destructive(policy, call, ctx)
    assert "mass.unbounded" in risk.matched_rules


def test_select_all_alone_is_only_a_caution(policy):
    call, ctx = click("Select all")
    risk = not_destructive(policy, call, ctx)
    assert "mass.select_all" in risk.matched_rules


# --------------------------------------------------------------------------- #
# 10. false positives - the UI must stay usable
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "label,kw",
    [
        ("Search", {"url": f"{SITE}/search", "title": "Search"}),
        ("Search products", {"role": "textbox"}),
        ("Поиск", {"url": f"{SITE}/search", "title": "Поиск"}),
        ("Add to cart", {"url": f"{SITE}/product/1", "title": "Aeron Chair"}),
        ("В корзину", {"url": f"{SITE}/product/1", "title": "Товар"}),
        ("View order history", {"url": f"{SITE}/account", "title": "My account"}),
        ("Мои заказы", {"url": f"{SITE}/account", "title": "Аккаунт"}),
        ("Показать все заказы", {"url": f"{SITE}/account", "title": "Аккаунт"}),
        ("Next page", {"url": f"{SITE}/search", "title": "Results"}),
        ("Print", {"url": f"{SITE}/account", "title": "Account"}),
        ("Got it", {"title": "Shop"}),
        ("Continue shopping", {"url": f"{SITE}/cart", "title": "Your cart"}),
        ("Add to favourites", {"url": f"{SITE}/product/1", "title": "Chair"}),
        ("Скачать прайс", {"role": "link", "element_kw": {"href": f"{SITE}/price.pdf"}}),
    ],
)
def test_false_positives_are_never_destructive(policy, label, kw):
    call, ctx = click(label, **kw)
    not_destructive(policy, call, ctx)


@pytest.mark.parametrize("label", ["Search", "Add to cart", "View order history", "Continue shopping"])
def test_false_positives_do_not_require_confirmation(policy, label):
    call, ctx = click(label)
    risk = policy.classify(call, ctx)
    assert risk.requires_confirmation is False
    assert "destructive" not in risk.level


# --------------------------------------------------------------------------- #
# 11. meaning, not site knowledge (the anti-hardcoding invariant)
# --------------------------------------------------------------------------- #

def test_policy_flags_unseen_checkout_flow_by_meaning(make_policy):
    """A site the author invented must still be gated correctly."""
    policy = make_policy()
    site, title = "https://totally-unknown-store.test", "Оформление заказа"
    remembered = [
        click("Place order", url=f"{site}/final-step", title=title),
        click("Оплатить", url=f"{site}/final-step", title=title),
        click("Удалить аккаунт", url=f"{site}/profile", title="Профиль"),
        click("Send", url=f"{site}/inbox", title="Inbox"),
    ]
    for call, ctx in remembered:
        destructive(policy, call, ctx)

    # nothing in the rule table may mention a real site
    blob = json.dumps([{"id": r.id, "reason": r.reason, "pattern": r.pattern.pattern if r.pattern else ""}
                       for r in RULES], ensure_ascii=False).lower()
    for domain in ("example", "google", "facebook", "hh.ru", "amazon", "ozon"):
        assert domain not in blob, f"rule table leaks site knowledge: {domain}"


def test_classification_does_not_depend_on_the_host(policy):
    calls = []
    for host in ("https://a.test", "https://b.test", "https://в-кириллице.test"):
        call, ctx = click("Удалить аккаунт", url=f"{host}/settings", title="Настройки")
        calls.append(policy.classify(call, ctx))
    assert {c.level for c in calls} == {"destructive"}
    assert {tuple(c.matched_rules) for c in calls} == {tuple(calls[0].matched_rules)}


def test_rule_table_is_well_formed():
    ids = [r.id for r in default_rules()]
    assert len(ids) == len(set(ids)), "rule ids must be unique"
    for rule in RULES:
        assert rule.id and "." in rule.id
        assert rule.reason
        assert rule.level in ("safe", "caution", "destructive")
        assert rule.level != "safe", "a rule that does nothing should not exist"
        assert rule.pattern is not None or rule.guard is not None


def test_classify_is_total_on_hostile_input(policy):
    weird = [
        ToolCall("totally_unknown_tool", {}),
        ToolCall("click", {}),
        ToolCall("click", {"element_id": "not-an-int", "intent": None}),
        ToolCall("type_text", {"element_id": 0, "text": None}),
        ToolCall("goto", {"url": "not a url at all"}),
        ToolCall("tabs", {"action": "close"}),
    ]
    for call in weird:
        risk = policy.classify(call, ActionContext())
        assert risk.level in ("safe", "caution", "destructive")
    # deterministic: same input, same verdict
    call, ctx = click("Place order", url=f"{SITE}/checkout", title="Checkout")
    assert policy.classify(call, ctx) == policy.classify(call, ctx)


def test_classification_is_deterministic_across_two_policies(make_policy):
    a, b = make_policy(), make_policy()
    call, ctx = click("Оплатить заказ", url=f"{SITE}/checkout", title="Оформление")
    assert a.classify(call, ctx) == b.classify(call, ctx)


# --------------------------------------------------------------------------- #
# 12. authorize(): confirm modes, allow-list, password policy
# --------------------------------------------------------------------------- #

def destructive_call():
    return click("Place order", url=f"{SITE}/checkout", title="Checkout")


def test_ask_mode_approves_when_the_human_says_yes(policy):
    call, ctx = destructive_call()
    seen: list[tuple[str, str]] = []

    def confirm(prompt: str, details: str) -> bool:
        seen.append((prompt, details))
        return True

    approved, reason = policy.authorize(call, policy.classify(call, ctx), ctx, confirm)
    assert approved is True
    assert len(seen) == 1
    prompt, details = seen[0]
    assert "Place order" in prompt
    assert "destructive" in details and "pay." in details
    assert "approved" in reason


def test_ask_mode_denies_when_the_human_says_no(policy):
    call, ctx = destructive_call()

    def confirm(prompt: str, details: str) -> bool:
        return False

    approved, reason = policy.authorize(call, policy.classify(call, ctx), ctx, confirm)
    assert approved is False
    assert "denied" in reason
    assert policy.audit_records()[-1]["decision"] == "denied_by_human"


def test_allow_mode_never_prompts_but_still_audits(make_policy):
    policy = make_policy(confirm_mode="allow")
    call, ctx = destructive_call()
    calls: list[str] = []

    approved, reason = policy.authorize(
        call, policy.classify(call, ctx), ctx, lambda p, d: calls.append(p) or True
    )
    assert approved is True and not calls
    assert "auto" in reason
    record = policy.audit_records()[-1]
    assert record["decision"] == "auto_approved" and record["human_answer"] is None


def test_yolo_mode_never_prompts_and_marks_the_audit_trail(make_policy):
    policy = make_policy(confirm_mode="yolo")
    call, ctx = destructive_call()
    calls: list[str] = []

    approved, reason = policy.authorize(
        call, policy.classify(call, ctx), ctx, lambda p, d: calls.append(p) or True
    )

    assert approved is True and not calls, "yolo must not ask the human"
    assert "yolo" in reason
    record = policy.audit_records()[-1]
    assert record["decision"] == "auto_approved_yolo" and record["human_answer"] is None


def test_yolo_mode_still_refuses_password_fields(make_policy):
    """The password rule is not a confirmation: yolo cannot switch it off."""
    policy = make_policy(confirm_mode="yolo")
    call, ctx = typing("Password", PASSWORD, field_type="password",
                       url=f"{SITE}/login", title="Sign in", password=True)

    approved, reason = policy.authorize(
        call, policy.classify(call, ctx), ctx, lambda p, d: True
    )

    assert approved is False
    assert "human" in reason
    assert policy.audit_records()[-1]["decision"] == "refused"


def test_deny_mode_refuses_every_destructive_action(make_policy):
    policy = make_policy(confirm_mode="deny")
    call, ctx = destructive_call()
    calls: list[str] = []

    approved, reason = policy.authorize(
        call, policy.classify(call, ctx), ctx, lambda p, d: calls.append(p) or True
    )
    assert approved is False and not calls
    assert "deny" in reason
    assert policy.audit_records()[-1]["decision"] == "denied_by_config"


def test_allow_destructive_tools_false_refuses_even_in_allow_mode(make_policy):
    policy = make_policy(confirm_mode="allow", allow_destructive_tools=False)
    call, ctx = destructive_call()
    calls: list[str] = []
    approved, reason = policy.authorize(
        call, policy.classify(call, ctx), ctx, lambda p, d: calls.append(p) or True
    )
    assert approved is False and not calls
    assert "allow_destructive_tools" in reason


def test_safe_actions_are_approved_without_asking(policy):
    call, ctx = click("Search", url=f"{SITE}/search", title="Search")
    calls: list[str] = []
    approved, reason = policy.authorize(
        call, policy.classify(call, ctx), ctx, lambda p, d: calls.append(p) or True
    )
    assert approved is True and not calls
    assert "without confirmation" in reason
    assert policy.audit_records()[-1]["decision"] == "allowed_no_confirmation"


def test_caution_actions_are_approved_without_asking(policy):
    call, ctx = click("Checkout", role="link", url=f"{SITE}/cart", title="Your cart")
    risk = policy.classify(call, ctx)
    assert risk.level == "caution"
    calls: list[str] = []
    approved, _ = policy.authorize(call, risk, ctx, lambda p, d: calls.append(p) or True)
    assert approved is True and not calls


def test_auto_approve_domains_skips_the_prompt(make_policy):
    policy = make_policy(auto_approve_domains=["example.invalid"])
    call, ctx = destructive_call()          # ctx host == shop.example.invalid
    calls: list[str] = []
    approved, reason = policy.authorize(
        call, policy.classify(call, ctx), ctx, lambda p, d: calls.append(p) or True
    )
    assert approved is True and not calls
    assert "auto_approve_domains" in reason
    assert policy.audit_records()[-1]["decision"] == "auto_approved_domain"


@pytest.mark.parametrize(
    "url,domains,expect_allowed",
    [
        (f"{SITE}/checkout", ["example.invalid"], True),
        (f"{SITE}/checkout", ["shop.example.invalid"], True),
        (f"{SITE}/checkout", ["other.invalid"], False),
        # suffix matching must not be fooled by a lookalike host
        ("https://example.invalid.evil.test/checkout", ["example.invalid"], False),
        ("https://notexample.invalid/checkout", ["example.invalid"], False),
    ],
)
def test_auto_approve_domains_suffix_matching(make_policy, url, domains, expect_allowed):
    policy = make_policy(auto_approve_domains=domains)
    call, ctx = click("Place order", url=url, title="Checkout")
    calls: list[str] = []
    approved, _ = policy.authorize(
        call, policy.classify(call, ctx), ctx, lambda p, d: calls.append(p) or False
    )
    if expect_allowed:
        assert approved is True and not calls, "an allow-listed host must not prompt"
    else:
        assert approved is False and len(calls) == 1, "a non-listed host must prompt"


def test_host_matches_helper():
    assert host_matches("shop.example.com", ["example.com"])
    assert host_matches("example.com", ["example.com"])
    assert not host_matches("example.com.evil.net", ["example.com"])
    assert not host_matches("notexample.com", ["example.com"])
    assert not host_matches("", ["example.com"])


def test_password_typing_is_refused_when_disabled(make_policy):
    policy = make_policy(allow_password_typing=False, confirm_mode="allow")
    call, ctx = typing("Password", PASSWORD, field_type="password",
                       url=f"{SITE}/login", title="Sign in", password=True)
    calls: list[str] = []
    approved, reason = policy.authorize(
        call, policy.classify(call, ctx), ctx, lambda p, d: calls.append(p) or True
    )
    assert approved is False and not calls
    assert "password" in reason.lower()
    assert policy.audit_records()[-1]["decision"] == "refused"


def test_password_typing_goes_through_the_human_when_enabled(make_policy):
    policy = make_policy(allow_password_typing=True, confirm_mode="ask")
    call, ctx = typing("Password", PASSWORD, field_type="password", password=True)
    calls: list[str] = []
    approved, _ = policy.authorize(
        call, policy.classify(call, ctx), ctx, lambda p, d: calls.append(p) or True
    )
    assert approved is True and len(calls) == 1


def test_a_broken_confirm_callback_is_treated_as_a_denial(policy):
    call, ctx = destructive_call()

    def boom(prompt: str, details: str) -> bool:
        raise RuntimeError("ui exploded")

    approved, reason = policy.authorize(call, policy.classify(call, ctx), ctx, boom)
    assert approved is False
    assert "prompt failed" in reason


def test_authorize_never_leaks_the_password_into_the_confirm_panel(make_policy):
    policy = make_policy(allow_password_typing=True)
    call, ctx = typing("Password", PASSWORD, field_type="password", password=True)
    seen: list[str] = []
    policy.authorize(call, policy.classify(call, ctx), ctx,
                     lambda p, d: seen.append(p + "\n" + d) or True)
    assert seen, "the human must be asked when password typing is enabled"
    assert PASSWORD not in seen[0]


# --------------------------------------------------------------------------- #
# 13. the LLM second opinion (optional, caution-only, must degrade)
# --------------------------------------------------------------------------- #

def caution_call(site: str = SITE):
    """'Checkout' link on a cart page: deterministic verdict is *caution*."""
    return click("Checkout", role="link", url=f"{site}/cart", title="Your cart")


def test_llm_is_not_called_when_the_check_is_off(make_policy):
    llm = ScriptedLLM('{"level": "destructive", "reason": "nope"}')
    policy = make_policy(llm=llm, llm_risk_check=False)
    call, ctx = caution_call()
    risk = policy.classify(call, ctx)
    assert risk.level == "caution"
    assert llm.calls == []


def test_llm_is_not_called_for_safe_actions(make_policy):
    llm = ScriptedLLM('{"level": "destructive", "reason": "nope"}')
    policy = make_policy(llm=llm, llm_risk_check=True)
    call, ctx = click("Search", url=f"{SITE}/search", title="Search")
    assert policy.classify(call, ctx).level == "safe"
    assert llm.calls == []


def test_llm_escalates_a_caution_to_destructive(make_policy):
    llm = ScriptedLLM('{"level": "destructive", "reason": "commits a payment"}')
    policy = make_policy(llm=llm, llm_risk_check=True)
    call, ctx = caution_call()
    risk = policy.classify(call, ctx)
    assert risk.level == "destructive"
    assert "llm.escalated" in risk.matched_rules
    assert llm.calls and llm.calls[0]["temperature"] == 0.0


def test_llm_error_keeps_the_deterministic_verdict(make_policy):
    llm = ScriptedLLM(error=LLMError("502 upstream", status=502))
    policy = make_policy(llm=llm, llm_risk_check=True)
    call, ctx = caution_call()
    risk = policy.classify(call, ctx)
    assert risk.level == "caution"
    assert any("llm.degraded" in r for r in risk.reasons)


@pytest.mark.parametrize("garbage", ["", "I think it is fine", "{not json}", "[]", '{"level": "maybe"}',
                                     '{"verdict": "destructive"}'])
def test_unparsable_llm_answer_keeps_the_deterministic_verdict(make_policy, garbage):
    policy = make_policy(llm=ScriptedLLM(garbage), llm_risk_check=True)
    call, ctx = caution_call()
    risk = policy.classify(call, ctx)
    assert risk.level == "caution"
    assert any("llm.degraded" in r for r in risk.reasons)


def test_missing_llm_keeps_the_deterministic_verdict(make_policy):
    policy = make_policy(llm=None, llm_risk_check=True)
    call, ctx = caution_call()
    assert policy.classify(call, ctx).level == "caution"


def test_fenced_json_is_parsed(make_policy):
    llm = ScriptedLLM('```json\n{"level": "destructive", "reason": "pay"}\n```')
    policy = make_policy(llm=llm, llm_risk_check=True)
    assert policy.classify(*caution_call()).level == "destructive"


# --------------------------------------------------------------------------- #
# 14. audit trail and redaction
# --------------------------------------------------------------------------- #

def test_audit_file_is_jsonl_with_the_expected_fields(policy, tmp_path: Path):
    call, ctx = destructive_call()
    policy.authorize(call, policy.classify(call, ctx), ctx, lambda p, d: True)
    path = Path(policy.config.audit_path)
    assert path.exists()
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1
    record = json.loads(lines[0])
    for key in ("ts", "tool", "args", "url", "title", "element", "risk", "rules",
                "reasons", "decision", "approved", "human_answer", "confirm_mode", "reason"):
        assert key in record, key
    assert record["tool"] == "click"
    assert record["risk"] == "destructive"
    assert record["approved"] is True
    assert record["human_answer"] is True
    assert record["url"] == f"{SITE}/checkout"


def test_audit_is_append_only(policy):
    for _ in range(3):
        call, ctx = destructive_call()
        policy.authorize(call, policy.classify(call, ctx), ctx, lambda p, d: True)
    assert len(policy.audit_records()) == 3
    assert [r["decision"] for r in policy.audit_records()] == ["approved_by_human"] * 3


def test_audit_never_contains_a_password(policy):
    call, ctx = typing("Password", PASSWORD, field_type="password",
                       url=f"{SITE}/login", title="Sign in", password=True)
    risk = policy.classify(call, ctx)
    policy.authorize(call, risk, ctx, lambda p, d: True)
    raw = Path(policy.config.audit_path).read_text(encoding="utf-8")
    assert PASSWORD not in raw
    assert PASSWORD not in json.dumps(policy.audit_records(), ensure_ascii=False)
    assert REDACTED in raw
    assert policy.audit_records()[-1]["args"]["text"] == REDACTED


def test_audit_redacts_secret_looking_values_under_innocent_keys(policy):
    for secret in (CARD, API_KEY, JWT):
        call = ToolCall("type_text", {"element_id": 3, "text": secret, "submit": False})
        ctx = ActionContext(url=SITE, page_title="Note",
                            element=element("textbox", "Note", type="text"))
        policy.authorize(call, policy.classify(call, ctx), ctx, lambda p, d: True)
    raw = Path(policy.config.audit_path).read_text(encoding="utf-8")
    for secret in (CARD, API_KEY, JWT):
        assert secret not in raw
        assert secret.replace(" ", "") not in raw
    assert "4111" not in raw


def test_redaction_helpers():
    assert looks_secret(CARD) and looks_secret(API_KEY) and looks_secret(JWT)
    assert not looks_secret("office chair") and not looks_secret("https://example.com/a/b")
    masked = redact_secrets(f"card {CARD} key {API_KEY}")
    assert CARD not in masked and API_KEY not in masked
    args = redact_args({"text": "hello", "password": PASSWORD, "n": 3})
    assert args["password"] == REDACTED and args["text"] == "hello" and args["n"] == 3
    assert redact_args({"text": PASSWORD}, password_target=True)["text"] == REDACTED


def test_audit_failure_never_crashes_the_run(make_policy, tmp_path: Path):
    """An unwritable audit file must be recorded, not raised into the run."""
    policy = make_policy()
    blocker = tmp_path / "audit-dir-in-the-way"
    blocker.write_text("not a directory", encoding="utf-8")
    policy.config.audit_path = blocker / "audit.jsonl"   # a file where a dir is needed
    call, ctx = destructive_call()
    approved, _ = policy.authorize(call, policy.classify(call, ctx), ctx, lambda p, d: True)
    assert approved is True            # the decision still happened
    assert policy.audit_errors         # ... and the failure was recorded


def test_audit_records_is_empty_when_the_file_does_not_exist(policy):
    assert policy.audit_records() == []
