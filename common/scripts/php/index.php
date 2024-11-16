<html>
<head>
<title><?php echo getcwd(); ?></title>
<style type='text/css'>
body {
    font-family: "Helvetica", sans-serif;
    font-size: 9pt;
    line-height: 10.5pt;
}
h1 {
    font-size: 14pt;
    margin: 0.5em 1em 0.2em 1em;
    text-align: left;
}
div.pic h3 { 
    font-size: 11pt;
    margin: 0.5em 1em 0.2em 1em;
}
div.pic p {
    font-size: 11pt;
    margin: 0.0em 1em 0.2em 1em;
}
div.pic {
    display: block;
    float: left;
    background-color: white;
    border: 1px solid #ccc;
    padding: 2px;
    text-align: center;
    margin: 40px 10px 10px 2px;
   /* -moz-box-shadow: 7px 5px 5px rgb(80,80,80);    /* Firefox 3.5 */
   /* -webkit-box-shadow: 7px 5px 5px rgb(80,80,80); /* Chrome, Safari */
   /* box-shadow: 7px 5px 5px rgb(80,80,80);         /* New browsers */  
}
div.list {
    font-size: 13pt;
    margin: 0.5em 1em 1.2em 1em;
    display: block; 
    clear: both;
}
div.list li {
    margin-top: 0.3em;
}
a { text-decoration: none; color: #29407C; }
a:hover { text-decoration: underline; color: #D08504; }
</style>
</head>
<body>
<h1><form>filter  <input type="text" name="match" size="30" value="<?php if (isset($_GET['match'])) print htmlspecialchars($_GET['match']);  ?>" /><input type="Submit" value="Go" /></form></h1>
<div>
<?php
$parent  = "../";
print "<h3><a class=\"file\" href=\"$parent\">Parent Directory</a></h3>"
?>
</div>
<div>
<?php
$displayed = array();
if ($_GET['noplots']) {
    print "Plots will not be displayed.\n";
} else {
    $other_exts = array('log.png', 'lin.pdf', 'lin.cxx', 'lin.eps', 'lin.root', 'lin.txt', 'log.pdf', 'log.cxx', 'log.eps', 'log.root', 'log.txt');
    $filenames = glob("*lin.png"); sort($filenames);
    $used = array();
    foreach ($filenames as $filename) {
        if (isset($_GET['match']) && !fnmatch('*'.$_GET['match'].'*', $filename)) continue;
        array_push($displayed, $filename);
        array_push($used,$filename);
        print "<div class='pic'>\n";
        print "<h3><a href=\"$filename\">$filename</a></h3>";
        print "<a href=\"$filename\"><img src=\"$filename\" style=\"border: none; width: 300px; \"></a>";
        $others = array();
        foreach ($other_exts as $ex) {
            $other_filename = str_replace('lin.png', $ex, $filename);
            if (file_exists($other_filename)) {
                array_push($used,$other_filename);
                array_push($others, "<a class=\"file\" href=\"$other_filename\">[" . $ex . "]</a>");
                if ($ex != '.txt') array_push($displayed, $other_filename);
            }
        }
        print "<p> Last changed: " . date ("F d Y H:i:s.", filemtime($filename)) . " </p>";
        if ($others) print "<p>Also as ".implode(', ',$others)."</p>";
        print "</div>";
    }
    $other_exts = array('.pdf', '.cxx', '.eps', '.root', '.txt');
    $filenames = glob("*.png"); sort($filenames);
    foreach ($filenames as $filename) {
        if (isset($_GET['match']) && !fnmatch('*'.$_GET['match'].'*', $filename)) continue;
        if (in_array($filename,$used)) continue;
        array_push($displayed, $filename);
        print "<div class='pic'>\n";
        print "<h3><a href=\"$filename\">$filename</a></h3>";
        print "<a href=\"$filename\"><img src=\"$filename\" style=\"border: none; width: 300px; \"></a>";
        $others = array();
        foreach ($other_exts as $ex) {
            $other_filename = str_replace('.png', $ex, $filename);
            if (file_exists($other_filename)) {
                array_push($others, "<a class=\"file\" href=\"$other_filename\">[" . $ex . "]</a>");
                if ($ex != '.txt') array_push($displayed, $other_filename);
            }
        }
        print "<p> Last changed: " . date ("F d Y H:i:s.", filemtime($filename)) . " </p>";
        if ($others) print "<p>Also as ".implode(', ',$others)."</p>";
        print "</div>";
    }
}
?>
</div>
<div class="list">
<ul>
<?php
$nondirs = array();
foreach (glob("*") as $filename) {
    if ($_GET['noplots'] || !in_array($filename, $displayed)) {
        if (isset($_GET['match']) && !fnmatch('*'.$_GET['match'].'*', $filename)) continue;
        if (is_dir($filename)) {
            print "<li>[DIR] <a href=\"$filename\">$filename</a></li>";
        } else if ($filename != "index.php") {
            array_push($nondirs,"<li><a href=\"$filename\">$filename</a></li>");
        }
    }
}
foreach ($nondirs as $file) {
    print $file;
}
?>
</ul>
</div>
<div>
<?php
$parent  = "../";
print "<h3><a class=\"file\" href=\"$parent\">Parent Directory</a></h3>"
?>
</div>
<h3>a gp production</h3>
</body>
</html>
