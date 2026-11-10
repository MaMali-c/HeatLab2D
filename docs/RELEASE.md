# Release and deployment · انتشار و رفع خطا

## Preflight

- [ ] Verify the repository root contains `app.py`, `requirements.txt`, `.streamlit/config.toml` and the font files in `assets/`.
- [ ] Install the pinned dependencies, run `python -m pip check`, then run `python -m pytest -q`.
- [ ] Generate a fresh benchmark PDF with `python -m heatlab.scientific_reporting --output validation-report`.
- [ ] Run the dashboard and exercise the default transient model, a steady model, the compare view and a PDF download.
- [ ] Push the release commit to the public GitHub repository and wait for both GitHub Actions jobs to pass.
- [ ] Deploy the release branch and verify the app URL in a signed-out browser.

## Streamlit Community Cloud

1. In the [Community Cloud workspace](https://share.streamlit.io/), select **Create app** and choose the public GitHub repository.
2. Set **Branch** to `main` and **Main file path** to `app.py`.
3. In **Advanced settings**, choose **Python 3.14**. Cloud reads `requirements.txt` from the repository root. [Deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) · [Dependency discovery](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
4. Deploy. Once the app loads, check its sharing settings and open the assigned URL in a signed-out window. Verify the temperature plot, one benchmark and a download. [Sharing instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app)

Base deployment supports PDF, CSV, JSON, PNG and GIF. MP4 is an optional local/CI feature enabled by `requirements-video.txt`. App inputs and exports are generated in each session, so the deployment does not need project API keys or a database.

## رفع خطا

| مشکل | اقدام |
| --- | --- |
| دستور `python3.14` پیدا نمی‌شود | نسخهٔ مناسب مک را از python.org نصب کنید و در پنجرهٔ جدید ترمینال `python3.14 --version` را بررسی کنید. |
| نصب بسته‌ها روی مک یا در Cloud شکست می‌خورد | نسخهٔ Python و معماری مفسر را بررسی کنید. در Cloud، نخستین خطای نصب را در **Manage app** بخوانید و از Python 3.14 و فایل وابستگی در ریشهٔ مخزن استفاده کنید. |
| دستگاه پایا با شرط شار حل نمی‌شود | برای چهار ضلع شار یا عایق، مجموع توان ورودی و تولید حجمی باید صفر باشد. برای بار سازگار، دمای میانگین مرجع را تعیین کنید. |
| اجرای ناپایا غیرفعال شده است | تعداد سلول‌ها یا گام‌های زمانی را تا سقف اعلام‌شده در رابط کاهش دهید. |
| MP4 در فهرست خروجی دیده نمی‌شود | محلی، `python -m pip install -r requirements-video.txt` را اجرا و برنامه را دوباره راه‌اندازی کنید. GIF از وابستگی‌های اصلی استفاده می‌کند. |
| لینک عمومی ورود می‌خواهد | وضعیت **Sharing** را در Cloud بررسی کنید و نشانی نمایش برنامه را در مرورگر بدون ورود باز کنید. |
| خطا پس از انتشار | گزارش Cloud و اجرای GitHub Actions را با آخرین commit تطبیق دهید. در صورت نیاز تغییر مشکل‌دار را با یک commit بازگشتی اصلاح کنید. |

## چک‌لیست فارسی

- [ ] آزمون‌ها، ساخت PDF و اجرای محلی را بررسی کنید.
- [ ] مخزن عمومی و موفقیت هر دو اجرای GitHub Actions را بررسی کنید.
- [ ] در Cloud، فایل `app.py` و Python 3.14 را انتخاب کنید.
- [ ] لینک نهایی را بدون ورود باز کنید و حل نمونه، نمودار و دانلود PDF را آزمایش کنید.
