zip:
	rm -f airliftsolution.zip
	rm -rf airliftsolution
	mkdir airliftsolution
	mkdir airliftsolution/solution
	cp solution/mysolution.py airliftsolution/solution/
	cp postBuild airliftsolution/
	cp environment.yml airliftsolution/
	zip -r airliftsolution.zip airliftsolution/solution/mysolution.py airliftsolution/postBuild airliftsolution/environment.yml -x '**/.*' -x '**/__MACOSX'
	rm -rf airliftsolution