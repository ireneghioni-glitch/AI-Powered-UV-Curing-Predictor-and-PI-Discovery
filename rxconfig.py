import reflex as rx

config = rx.Config(
    app_name="app",
    db_url="sqlite:///app.db",
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.TailwindV4Plugin(),
        rx.plugins.RadixThemesPlugin(),
    ]
)