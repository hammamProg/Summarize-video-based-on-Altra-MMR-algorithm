run : app.py
	@python3 app.py

# @echo "\n\n\n\tGraduation Video Summarization (Video-MMR)" | boxes -d diamonds
pre-run : 
	@python3 -m venv venv
	@source venv/bin/activate
	@pip3 install -r requirements.txt
	@echo "Ready to run the app (make run)."