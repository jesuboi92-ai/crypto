"""Realtime crypto market monitor with Telegram alerts.

Tracks the 50 largest Binance Spot USDT markets by 24-hour quote volume.
Public market data requires no Binance API key.
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import requests
from dotenv import load_dotenv

load_dotenv()

TOP_MARKETS = 50
QUOTE_CURRENCY = "USDT"
TIMEFRAME = "1h"
RSI_PERIOD = 14
VOLUME_LOOKBACK = 20
VOLUME_SPIKE_MULTIPLIER = 2.5
PRICE_CHANGE_THRESHOLD = 5.0
ALERT_COOLDOWN_SECONDS = 6 * 60 * 60
CHECK_INTERVAL_MINUTES = max(1, int(os.getenv("CHECK_INTERVAL_MINUTES", "15")))
# Live orders are opt-in. The small defaults limit exposure per order and in total.
SIMULATION_SETTING = os.getenv("SIMULATION_MODE", "True").strip().lower()
if SIMULATION_SETTING not in {"1", "0", "true", "false", "yes", "no", "on", "off"}:
    raise ValueError("SIMULATION_MODE must be True or False.")
SIMULATION_MODE = SIMULATION_SETTING in {"1", "true", "yes", "on"}
ORDER_BUDGET_USDT = float(os.getenv("ORDER_BUDGET_USDT", "10"))
MAX_ORDER_USDT = float(os.getenv("MAX_ORDER_USDT", "10"))
MAX_OPEN_POSITIONS = max(1, int(os.getenv("MAX_OPEN_POSITIONS", "3")))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

# The bot only sells amounts it has recorded after its own buys.
STATE_FILE = Path(__file__).with_name("positions.json")
positions: dict[str, dict[str, float]] = {}
pending_orders: set[str] = set()

# Keep alerts in memory so the same signal does not repeat every polling cycle.
last_alerted: dict[tuple[str, str], float] = {}


def calculate_rsi(closes: list[float], period: int = RSI_PERIOD) -> float | None:
    """Calculate Wilder's RSI from closing prices; return None if history is short."""
    if len(closes) <= period:
        return None

    changes = [closes[index] - closes[index - 1] for index in range(1, len(closes))]
    gains = [max(change, 0.0) for change in changes]
    losses = [max(-change, 0.0) for change in changes]

    average_gain = sum(gains[:period]) / period
    average_loss = sum(losses[:period]) / period

    for index in range(period, len(changes)):
        average_gain = ((average_gain * (period - 1)) + gains[index]) / period
        average_loss = ((average_loss * (period - 1)) + losses[index]) / period

    if average_loss == 0:
        return 100.0 if average_gain > 0 else 50.0

    relative_strength = average_gain / average_loss
    return 100.0 - (100.0 / (1.0 + relative_strength))


def send_telegram_message(message: str) -> bool:
    """Send a message with the Telegram Bot API."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram is not configured; alert will only be printed to the console:")
        print(message)
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        response = requests.post(
            url,
            json={"chat_id": TELEGRAM_CHAT_ID, "text": message},
            timeout=15,
        )
        response.raise_for_status()
        if not response.json().get("ok"):
            print("Telegram rejected the message.")
            return False
        return True
    except requests.RequestException:
        # Do not print the exception: its URL may contain the bot token.
        print("Telegram delivery failed due to a network error.")
        return False


def maybe_alert(symbol: str, signal: str, message: str) -> None:
    """Send a signal only once per cooldown window."""
    now = time.time()
    alert_key = (symbol, signal)
    last_time = last_alerted.get(alert_key, 0.0)
    if now - last_time < ALERT_COOLDOWN_SECONDS:
        return

    print(message)
    if send_telegram_message(message):
        last_alerted[alert_key] = now


def load_trade_state() -> None:
    """Restore bot-owned positions and unresolved orders from the local state file."""
    global positions, pending_orders
    if not STATE_FILE.exists():
        return

    state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    saved_simulation_mode = bool(state.get("simulation_mode", True))
    if saved_simulation_mode != SIMULATION_MODE:
        raise RuntimeError(
            "Trade state was created in a different simulation mode. "
            "Check the exchange and positions.json before changing modes."
        )
    positions = state.get("positions", {})
    pending_orders = set(state.get("pending_orders", []))


def save_trade_state() -> None:
    """Persist bot-owned positions safely across restarts."""
    state = {
        "simulation_mode": SIMULATION_MODE,
        "positions": positions,
        "pending_orders": sorted(pending_orders),
    }
    temporary_file = STATE_FILE.with_suffix(".tmp")
    temporary_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary_file.replace(STATE_FILE)


def notify_trade(side: str, symbol: str, amount: float, price: float) -> None:
    """Report a confirmed or simulated fill to Telegram and the console."""
    mode = "SIMULATION" if SIMULATION_MODE else "LIVE"
    message = (
        f"{mode} {side.upper()} {symbol}\n"
        f"Amount: {amount:.8g}\n"
        f"Average / estimated price: {price:.8g} USDT\n"
        f"Approximate value: {amount * price:.2f} USDT"
    )
    print(message)
    send_telegram_message(message)


def resolve_order(exchange: ccxt.binance, symbol: str, order: dict) -> dict:
    """Fetch an order once when the create-order response lacks fill details."""
    if order.get("filled") is not None and order.get("average") is not None:
        return order
    order_id = order.get("id")
    if not order_id:
        return order
    try:
        return exchange.fetch_order(order_id, symbol)
    except ccxt.BaseError:
        return order


def execute_market_buy(exchange: ccxt.binance, symbol: str, reference_price: float) -> None:
    """Buy one fixed USDT budget, or simulate it, if no bot position exists."""
    if symbol in positions or symbol in pending_orders:
        return
    if len(positions) + len(pending_orders) >= MAX_OPEN_POSITIONS:
        return
    if ORDER_BUDGET_USDT <= 0 or ORDER_BUDGET_USDT > MAX_ORDER_USDT:
        raise ValueError("ORDER_BUDGET_USDT must be positive and no greater than MAX_ORDER_USDT.")

    minimum_cost = exchange.market(symbol).get("limits", {}).get("cost", {}).get("min")
    if minimum_cost is not None and ORDER_BUDGET_USDT < minimum_cost:
        print(f"Buy skipped for {symbol}: order budget is below the exchange minimum.")
        return

    if SIMULATION_MODE:
        amount = float(exchange.amount_to_precision(symbol, ORDER_BUDGET_USDT / reference_price))
        if amount <= 0:
            print(f"Buy skipped for {symbol}: amount rounds down to zero.")
            return
        positions[symbol] = {"amount": amount, "entry_price": reference_price}
        save_trade_state()
        notify_trade("buy", symbol, amount, reference_price)
        return

    # CCXT interprets the market-buy amount as quote currency when this option is false.
    pending_orders.add(symbol)
    save_trade_state()
    try:
        order = exchange.create_market_buy_order(symbol, ORDER_BUDGET_USDT)
        order = resolve_order(exchange, symbol, order)
    except ccxt.BaseError as error:
        print(f"Buy result for {symbol} is unresolved ({type(error).__name__}); check the exchange before retrying.")
        send_telegram_message(f"Unresolved BUY order for {symbol}. Check the exchange before restarting or retrying.")
        return

    filled = order.get("filled")
    average_price = order.get("average") or order.get("price") or reference_price
    if filled is None or filled <= 0:
        print(f"Buy result for {symbol} has no confirmed fill; check the exchange before retrying.")
        send_telegram_message(f"BUY order for {symbol} has no confirmed fill. Check the exchange manually.")
        return

    positions[symbol] = {"amount": float(filled), "entry_price": float(average_price)}
    pending_orders.remove(symbol)
    save_trade_state()
    notify_trade("buy", symbol, float(filled), float(average_price))


def execute_market_sell(exchange: ccxt.binance, symbol: str, reference_price: float) -> None:
    """Sell only the amount recorded as purchased by this bot."""
    position = positions.get(symbol)
    if position is None or symbol in pending_orders:
        return

    amount = float(position["amount"])
    if amount <= 0:
        return
    if SIMULATION_MODE:
        positions.pop(symbol)
        save_trade_state()
        notify_trade("sell", symbol, amount, reference_price)
        return

    try:
        order_amount = float(exchange.amount_to_precision(symbol, amount))
    except ccxt.BaseError as error:
        print(f"Sell amount could not be formatted for {symbol} ({type(error).__name__}).")
        return
    if order_amount <= 0:
        print(f"Sell skipped for {symbol}: amount rounds down to zero.")
        return

    pending_orders.add(symbol)
    save_trade_state()
    try:
        order = exchange.create_market_sell_order(symbol, order_amount)
        order = resolve_order(exchange, symbol, order)
    except ccxt.BaseError as error:
        print(f"Sell result for {symbol} is unresolved ({type(error).__name__}); check the exchange before retrying.")
        send_telegram_message(f"Unresolved SELL order for {symbol}. Check the exchange before restarting or retrying.")
        return

    filled = order.get("filled")
    average_price = order.get("average") or order.get("price") or reference_price
    if filled is None or filled <= 0:
        print(f"Sell result for {symbol} has no confirmed fill; check the exchange before retrying.")
        send_telegram_message(f"SELL order for {symbol} has no confirmed fill. Check the exchange manually.")
        return

    remaining = max(0.0, amount - float(filled))
    if remaining > 0:
        position["amount"] = remaining
    else:
        positions.pop(symbol)
    pending_orders.remove(symbol)
    save_trade_state()
    notify_trade("sell", symbol, float(filled), float(average_price))


def get_top_markets(exchange: ccxt.binance) -> list[tuple[str, dict]]:
    """Return the top USDT spot pairs ranked by 24-hour quote volume."""
    markets = exchange.load_markets()
    tickers = exchange.fetch_tickers()
    candidates: list[tuple[str, dict]] = []

    for symbol, ticker in tickers.items():
        market = markets.get(symbol, {})
        if (
            not symbol.endswith(f"/{QUOTE_CURRENCY}")
            or market.get("active") is False
            or not (market.get("spot") or market.get("type") == "spot")
        ):
            continue

        quote_volume = ticker.get("quoteVolume")
        if quote_volume is None:
            continue
        candidates.append((symbol, ticker))

    candidates.sort(key=lambda item: item[1].get("quoteVolume") or 0, reverse=True)
    return candidates[:TOP_MARKETS]


def inspect_market(exchange: ccxt.binance, symbol: str, ticker: dict) -> None:
    """Check one market's 24-hour move, hourly RSI, and volume spike."""
    price_change = ticker.get("percentage")
    last_price = ticker.get("last")
    quote_volume = ticker.get("quoteVolume")
    if last_price is None or last_price <= 0:
        return

    try:
        candles = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=30)
    except ccxt.BaseError as error:
        print(f"OHLCV request failed for {symbol}: {error}")
        return

    # The final candle is still forming; use only completed hourly candles.
    completed_candles = candles[:-1]
    if len(completed_candles) < RSI_PERIOD + 1:
        return

    closes = [float(candle[4]) for candle in completed_candles]
    volumes = [float(candle[5]) for candle in completed_candles]
    rsi = calculate_rsi(closes)
    if rsi is None:
        return

    change_text = f"{price_change:+.2f}%" if price_change is not None else "unavailable"
    summary = (
        f"{symbol} | hinta {last_price:g} USDT | "
        f"24 h {change_text} | "
        f"24 h volyymi {quote_volume:,.0f} USDT | RSI(14, 1h) {rsi:.1f}"
    )
    print(summary)

    if price_change is not None and abs(price_change) >= PRICE_CHANGE_THRESHOLD:
        direction = "up" if price_change > 0 else "down"
        maybe_alert(
            symbol,
            "24h-change",
            f"{symbol}: price is {direction} {price_change:+.2f}% over the last 24 hours.\n{summary}",
        )

    if rsi > 70:
        maybe_alert(symbol, "RSI-overbought", f"Overbought: {summary}")
    elif rsi < 30:
        maybe_alert(symbol, "RSI-oversold", f"Oversold: {summary}")

    # Compare the most recent completed hour to the previous 20 completed hours.
    volume_spike = False
    if len(volumes) >= VOLUME_LOOKBACK + 1:
        previous_average = sum(volumes[-VOLUME_LOOKBACK - 1 : -1]) / VOLUME_LOOKBACK
        latest_volume = volumes[-1]
        if previous_average > 0 and latest_volume >= previous_average * VOLUME_SPIKE_MULTIPLIER:
            volume_spike = True
            ratio = latest_volume / previous_average
            maybe_alert(
                symbol,
                "volume-spike",
                f"Volume spike: {symbol} hourly volume is {ratio:.1f}x the previous "
                f"20-hour average.\n{summary}",
            )

    # Buy only when both conditions agree; sell only an existing bot-owned position.
    if rsi < 30 and volume_spike:
        execute_market_buy(exchange, symbol, float(last_price))
    elif rsi > 70:
        execute_market_sell(exchange, symbol, float(last_price))


def scan_market(exchange: ccxt.binance) -> None:
    """Fetch the top markets and evaluate each one."""
    try:
        top_markets = get_top_markets(exchange)
    except ccxt.BaseError as error:
        print(f"Market data request failed: {error}")
        return

    if not top_markets:
        print("No USDT spot markets were found.")
        return

    print(
        f"[{datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}] "
        f"Checking {len(top_markets)} markets."
    )
    for symbol, ticker in top_markets:
        try:
            inspect_market(exchange, symbol, ticker)
        except (TypeError, ValueError, IndexError) as error:
            print(f"Data processing failed for {symbol}: {error}")


def main() -> None:
    """Run the monitor until interrupted."""
    if ORDER_BUDGET_USDT <= 0 or ORDER_BUDGET_USDT > MAX_ORDER_USDT:
        raise ValueError("ORDER_BUDGET_USDT must be positive and no greater than MAX_ORDER_USDT.")
    if not SIMULATION_MODE and (not API_KEY or not API_SECRET):
        raise RuntimeError("Live trading requires API_KEY and API_SECRET in the .env file.")

    load_trade_state()
    exchange = ccxt.binance(
        {
            "apiKey": API_KEY,
            "secret": API_SECRET,
            "enableRateLimit": True,
            "options": {
                "defaultType": "spot",
                "createMarketBuyOrderRequiresPrice": False,
            },
        }
    )
    print(
        f"Starting monitor: top {TOP_MARKETS} Binance Spot USDT pairs, "
        f"checking every {CHECK_INTERVAL_MINUTES} minutes. "
        f"Simulation mode: {SIMULATION_MODE}. "
        f"Budget per buy: {ORDER_BUDGET_USDT:g} USDT; max open positions: {MAX_OPEN_POSITIONS}."
    )
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to enable Telegram alerts.")

    try:
        while True:
            scan_market(exchange)
            time.sleep(CHECK_INTERVAL_MINUTES * 60)
    except KeyboardInterrupt:
        print("Monitor stopped.")
    finally:
        exchange.close()


if __name__ == "__main__":
    main()
