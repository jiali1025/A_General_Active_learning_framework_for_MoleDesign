#!/usr/bin/perl -w

open(OUT,">g09-result.xyz") or die ":( g09-result.xyz";

if ($ARGV[0] eq ""){
  die " Use: g092xyz <g09_output.log> \n\n"
}
if (!-s "$ARGV[0]"){
  die "File $ARGV[0] does not exist or is empty.\n\n";
}
$filein=$ARGV[0];

open(IN,"$filein") or die ":( $filein";
while(<IN>){
  if (/Standard orientation:/){
    read_geom();
    last;
  }
}
$nat=$g[0];
close(IN);

open(IN,"$filein") or die ":( $filein";
while(<IN>){
  if (/Standard orientation:/){
    undef($_);
    read_geom();
    
  }
}
print OUT "$nat\n\n";
for ($i=0;$i<=$nat-1;$i++){
  printf OUT " %3s  %14.8f  %14.8f  %14.8f\n",$s[$i],$x[$i],$y[$i],$z[$i];
}
close(IN);

close(OUT);

# =======================================================================

sub read_geom{
  my ($i);
  while(<IN>){
    if (/---/){
      $_=<IN>;
      $_=<IN>;
      $_=<IN>;
      undef($_);
      $i=0;
      while(<IN>){
         if (/---/){
           last;
         }else{ 
           chomp;$_ =~ s/^\s*//;$_ =~ s/\s*$//;
           (@g)=split(/\s+/,$_);
           $grb[$i]     =$g[0];
           $a_number[$i]=$g[1];
           $grb[$i]     =$g[2];
           $x[$i]       =$g[3];
           $y[$i]       =$g[4];
           $z[$i]       =$g[5];
           $s[$i]=symbol($a_number[$i]);
           $i++;
         }
      }
      last;
    }
  }
}

# =======================================================================

sub symbol{
  my ($an,$s);
  ($an)=@_;
  if ($an == 1){
    $s="H";
  }
  if ($an == 5){
    $s="B";
  }
  if ($an == 6){
    $s="C";
  }
  if ($an == 7){
    $s="N";
  }
  if ($an == 8){
    $s="O";
  }
  if ($an == 9){
    $s="F";
  }
  if ($an == 14){
    $s="Si";
  }
  if ($an == 16){
    $s="S";
  }
  if ($an == 17){
    $s="Cl";
  }
  if($an == 35){
    $s="Br";
  }
 return $s;
}
